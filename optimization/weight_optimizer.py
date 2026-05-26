"""
Weight Optimizer for Pairwise Comparison Feedback System.

Directly optimizes weights to maximize the likelihood that the weighted
component scores correctly predict comparison outcomes. Uses the
Bradley-Terry probability model with Davidson extension for ties.

Usage::

    optimizer = WeightOptimizer(db_path)

    # Single-shot direct optimization
    result = optimizer.optimize_weights_direct(category='others')

    # With cross-validation for robustness
    result = optimizer.optimize_weights_with_cv(category='others')

    # Apply optimized weights to config
    optimizer.apply_optimized_weights(result['new_weights'], category='others')

CLI::

    python facet.py --optimize-weights
"""

import json
import logging
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize
from db import DEFAULT_DB_PATH, get_connection

logger = logging.getLogger("facet.optimizer")


class WeightOptimizer:
    """
    Optimizes scoring weights from pairwise comparisons.

    Uses direct preference optimization: weights are chosen to maximize the
    Bradley-Terry/Davidson likelihood of the observed comparison outcomes
    given the per-photo weighted score (no intermediate "learned scores"
    table; raw comparisons feed the optimizer directly).
    """

    # Score components that can be weighted (must match photos table columns)
    SCORE_COMPONENTS = [
        # Primary quality metrics
        'aesthetic',
        'quality_score',
        'face_quality',
        'face_sharpness',
        'eye_sharpness',
        'tech_sharpness',
        # Composition metrics
        'comp_score',
        'power_point_score',
        'leading_lines_score',
        # Technical metrics
        'exposure_score',
        'color_score',
        'contrast_score',
        'dynamic_range_stops',
        'mean_saturation',
        'noise_sigma',          # Note: lower is better (inverted in scoring)
        # Bonuses
        'isolation_bonus',
    ]

    # Maximum natural scale for each feature, used to normalize to 0–1 before
    # optimization so that weight percentages are directly interpretable.
    FEATURE_SCALES = {
        'aesthetic': 10.0,
        'quality_score': 10.0,
        'face_quality': 10.0,
        'face_sharpness': 100.0,
        'eye_sharpness': 10.0,
        'tech_sharpness': 10.0,
        'comp_score': 10.0,
        'power_point_score': 10.0,
        'leading_lines_score': 10.0,
        'exposure_score': 10.0,
        'color_score': 10.0,
        'contrast_score': 10.0,
        'dynamic_range_stops': 15.0,
        'mean_saturation': 1.0,
        'noise_sigma': 5.0,
        'isolation_bonus': 10.0,
    }

    def __init__(self, db_path: str = DEFAULT_DB_PATH, config_path: str = 'scoring_config.json'):
        self.db_path = db_path
        self.config_path = config_path

    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize feature columns to 0–1 using FEATURE_SCALES.

        Each column is divided by its scale so that all features occupy roughly
        the same range.  This makes optimized weight percentages directly
        comparable across features with different natural units.
        """
        scales = np.array([
            self.FEATURE_SCALES.get(c, 10.0) for c in self.SCORE_COMPONENTS
        ])
        # Avoid division by zero
        scales = np.where(scales > 1e-8, scales, 1.0)
        return X / scales

    def optimize_weights_direct(
        self,
        category: Optional[str] = None,
        min_comparisons: int = 30,
        include_ties: bool = True,
        tie_sensitivity: float = 0.1,
        min_improvement_threshold: float = 2.0,
        l2_regularization: float = 0.01,
    ) -> Dict:
        """
        Directly optimize weights to maximize comparison agreement.

        Uses Bradley-Terry probability model:
            P(A > B) = sigmoid(score_A - score_B)

        With Davidson extension for ties:
            P(tie) = 2 * theta * sqrt(P(A) * P(B)) / (P(A) + P(B) + 2*theta*sqrt(P(A)*P(B)))

        This approach is superior to the two-stage method because:
        - Uses raw comparison data directly (no information loss)
        - Optimizes for actual prediction accuracy, not MSE to arbitrary scores
        - Handles ties properly with Davidson model
        - More data-efficient (works with fewer comparisons)

        Args:
            category: Optimize weights for specific category (or all if None)
            min_comparisons: Minimum comparisons required before optimization
            include_ties: Whether to include tie comparisons in optimization
            tie_sensitivity: Davidson theta parameter (higher = more ties expected)
            min_improvement_threshold: Only suggest changes if accuracy improves by this %
            l2_regularization: L2 penalty on weight changes from current weights

        Returns:
            Dict with:
            - old_weights, new_weights: weight dictionaries
            - accuracy_before, accuracy_after: % of comparisons predicted correctly
            - log_likelihood: final negative log-likelihood
            - suggest_changes: whether to apply the new weights
            - per_comparison: list of (photo_a, photo_b, winner, predicted_correct)
        """
        with get_connection(self.db_path) as conn:
            # Get comparisons
            where_clauses = ["winner IN ('a', 'b')"]
            if include_ties:
                where_clauses = ["winner IN ('a', 'b', 'tie')"]

            params = []
            if category:
                where_clauses.append("c.category = ?")
                params.append(category)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(f"""
                SELECT c.photo_a_path, c.photo_b_path, c.winner,
                       p1.aesthetic as a_aesthetic, p1.quality_score as a_quality_score,
                       p1.face_quality as a_face_quality, p1.face_sharpness as a_face_sharpness,
                       p1.eye_sharpness as a_eye_sharpness, p1.tech_sharpness as a_tech_sharpness,
                       p1.comp_score as a_comp_score, p1.power_point_score as a_power_point_score,
                       p1.leading_lines_score as a_leading_lines_score,
                       p1.exposure_score as a_exposure_score, p1.color_score as a_color_score,
                       p1.contrast_score as a_contrast_score, p1.dynamic_range_stops as a_dynamic_range_stops,
                       p1.mean_saturation as a_mean_saturation, p1.noise_sigma as a_noise_sigma,
                       p1.isolation_bonus as a_isolation_bonus,
                       p2.aesthetic as b_aesthetic, p2.quality_score as b_quality_score,
                       p2.face_quality as b_face_quality, p2.face_sharpness as b_face_sharpness,
                       p2.eye_sharpness as b_eye_sharpness, p2.tech_sharpness as b_tech_sharpness,
                       p2.comp_score as b_comp_score, p2.power_point_score as b_power_point_score,
                       p2.leading_lines_score as b_leading_lines_score,
                       p2.exposure_score as b_exposure_score, p2.color_score as b_color_score,
                       p2.contrast_score as b_contrast_score, p2.dynamic_range_stops as b_dynamic_range_stops,
                       p2.mean_saturation as b_mean_saturation, p2.noise_sigma as b_noise_sigma,
                       p2.isolation_bonus as b_isolation_bonus
                FROM comparisons c
                JOIN photos p1 ON c.photo_a_path = p1.path
                JOIN photos p2 ON c.photo_b_path = p2.path
                WHERE {where_sql}
            """, params)

            comparisons = []
            features_a_list = []
            features_b_list = []

            for row in cursor:
                winner = row['winner']
                features_a = [
                    float(row[f'a_{c}'] or 0.0) for c in self.SCORE_COMPONENTS
                ]
                features_b = [
                    float(row[f'b_{c}'] or 0.0) for c in self.SCORE_COMPONENTS
                ]
                comparisons.append({
                    'photo_a': row['photo_a_path'],
                    'photo_b': row['photo_b_path'],
                    'winner': winner,
                })
                features_a_list.append(features_a)
                features_b_list.append(features_b)

            if len(comparisons) < min_comparisons:
                return {
                    'error': f'Need at least {min_comparisons} comparisons (have {len(comparisons)})',
                    'comparison_count': len(comparisons),
                }

            X_a = self._scale_features(np.array(features_a_list))
            X_b = self._scale_features(np.array(features_b_list))
            n_features = len(self.SCORE_COMPONENTS)

            # Load current weights
            old_weights = self._load_current_weights(category)
            old_w = np.array([old_weights.get(c, 1.0/n_features) for c in self.SCORE_COMPONENTS])
            if old_w.sum() > 0:
                old_w = old_w / old_w.sum()
            else:
                old_w = np.ones(n_features) / n_features

            # Encode winners: 1 for 'a', -1 for 'b', 0 for 'tie'
            winners = []
            for comp in comparisons:
                if comp['winner'] == 'a':
                    winners.append(1)
                elif comp['winner'] == 'b':
                    winners.append(-1)
                else:
                    winners.append(0)
            winners = np.array(winners)

            theta = tie_sensitivity

            def neg_log_likelihood(weights, return_predictions=False):
                """Compute negative log-likelihood of comparison outcomes."""
                # Normalize weights
                w_sum = weights.sum()
                if w_sum > 1e-8:
                    w = weights / w_sum
                else:
                    w = np.ones(n_features) / n_features

                # Compute weighted scores
                scores_a = X_a @ w
                scores_b = X_b @ w
                diff = scores_a - scores_b

                total_nll = 0.0
                predictions = []

                for i, (d, winner) in enumerate(zip(diff, winners)):
                    if winner == 1:  # A wins
                        # Log P(A > B) = log(sigmoid(diff)) = -log(1 + exp(-diff))
                        if d > 20:
                            nll = 0.0
                        elif d < -20:
                            nll = -d
                        else:
                            nll = np.log1p(np.exp(-d))
                        total_nll += nll
                        pred_correct = d > 0
                    elif winner == -1:  # B wins
                        # Log P(B > A) = log(sigmoid(-diff)) = -log(1 + exp(diff))
                        if d < -20:
                            nll = 0.0
                        elif d > 20:
                            nll = d
                        else:
                            nll = np.log1p(np.exp(d))
                        total_nll += nll
                        pred_correct = d < 0
                    else:  # Tie - Davidson model approximation
                        # For ties, we use a simpler model: higher probability near diff=0
                        # -log P(tie) ≈ (diff/theta)^2 for small theta
                        # This encourages equal scores for ties
                        nll = (d / (theta + 0.1)) ** 2
                        total_nll += nll
                        pred_correct = abs(d) < 0.5  # Consider "correct" if scores close

                    if return_predictions:
                        predictions.append(pred_correct)

                # Add L2 regularization to discourage large changes from current weights
                l2_penalty = l2_regularization * np.sum((weights - old_w) ** 2)
                total_nll += l2_penalty

                if return_predictions:
                    return total_nll, predictions
                return total_nll

            def compute_accuracy(weights):
                """Compute prediction accuracy for given weights."""
                w_sum = weights.sum()
                if w_sum > 1e-8:
                    w = weights / w_sum
                else:
                    w = np.ones(n_features) / n_features

                scores_a = X_a @ w
                scores_b = X_b @ w
                diff = scores_a - scores_b

                correct = 0
                total = 0
                for d, winner in zip(diff, winners):
                    if winner == 1:  # A should win
                        if d > 0:
                            correct += 1
                        total += 1
                    elif winner == -1:  # B should win
                        if d < 0:
                            correct += 1
                        total += 1
                    # Ties don't count toward accuracy

                return (correct / total * 100) if total > 0 else 0.0

            # Calculate accuracy with old weights
            accuracy_before = compute_accuracy(old_w)

            # Optimization bounds and constraints
            bounds = [(0.0, 0.60) for _ in range(n_features)]
            constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}

            # Multiple random restarts
            best_result = None
            best_nll = float('inf')
            n_restarts = 5

            starting_points = [old_w.copy(), np.ones(n_features) / n_features]
            np.random.seed(42)
            for _ in range(n_restarts - 2):
                starting_points.append(np.random.dirichlet(np.ones(n_features)))

            for start in starting_points:
                try:
                    result = minimize(
                        neg_log_likelihood,
                        start,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 500, 'ftol': 1e-9}
                    )

                    if result.fun < best_nll:
                        best_nll = result.fun
                        best_result = result.x.copy()
                except Exception:
                    continue

            if best_result is None:
                best_result = old_w.copy()

            # Ensure valid weights
            new_w = np.maximum(best_result, 0.0)
            if new_w.sum() > 0:
                new_w = new_w / new_w.sum()
            else:
                new_w = np.ones(n_features) / n_features

            # Calculate metrics with new weights
            accuracy_after = compute_accuracy(new_w)
            _, predictions = neg_log_likelihood(new_w, return_predictions=True)

            # Build per-comparison breakdown
            per_comparison = []
            for i, (comp, pred_correct) in enumerate(zip(comparisons, predictions)):
                per_comparison.append({
                    'photo_a': comp['photo_a'],
                    'photo_b': comp['photo_b'],
                    'winner': comp['winner'],
                    'predicted_correct': pred_correct,
                })

            # Determine if we should suggest changes
            accuracy_improvement = accuracy_after - accuracy_before
            suggest_changes = accuracy_improvement >= min_improvement_threshold

            # Convert to dicts
            new_weights = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, new_w)}

            # Log the run
            conn.execute("""
                INSERT INTO weight_optimization_runs
                (category, comparisons_used, old_weights, new_weights, mse_before, mse_after)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                category,
                len(comparisons),
                json.dumps(old_weights),
                json.dumps(new_weights),
                accuracy_before,  # Store accuracy in mse fields for now
                accuracy_after
            ))
            conn.commit()

            return {
                'old_weights': old_weights,
                'new_weights': new_weights,
                'accuracy_before': round(accuracy_before, 1),
                'accuracy_after': round(accuracy_after, 1),
                'improvement': round(accuracy_improvement, 1),
                'log_likelihood': round(-best_nll, 4),
                'suggest_changes': suggest_changes,
                'comparisons_used': len(comparisons),
                'ties_included': sum(1 for c in comparisons if c['winner'] == 'tie'),
                'per_comparison': per_comparison,
                'method': 'direct_preference_optimization',
            }

    def optimize_weights_with_cv(
        self,
        category: Optional[str] = None,
        n_folds: int = 5,
        min_comparisons: int = 30,
        include_ties: bool = True,
    ) -> Dict:
        """
        K-fold cross-validation for robust weight optimization.

        Splits comparisons into k folds, trains on k-1 folds, and evaluates
        on the held-out fold. Returns average weights and CV accuracy.

        Args:
            category: Category to optimize
            n_folds: Number of cross-validation folds
            min_comparisons: Minimum comparisons required
            include_ties: Whether to include ties

        Returns:
            Dict with:
            - average_weights: ensemble weights from all folds
            - cv_accuracy: mean accuracy on held-out comparisons
            - cv_std: standard deviation of accuracy across folds
            - fold_results: per-fold accuracy scores
        """
        with get_connection(self.db_path) as conn:
            # Get all comparisons
            where_clauses = ["winner IN ('a', 'b')"]
            if include_ties:
                where_clauses = ["winner IN ('a', 'b', 'tie')"]

            params = []
            if category:
                where_clauses.append("c.category = ?")
                params.append(category)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(f"""
                SELECT c.photo_a_path, c.photo_b_path, c.winner,
                       p1.aesthetic as a_aesthetic, p1.quality_score as a_quality_score,
                       p1.face_quality as a_face_quality, p1.face_sharpness as a_face_sharpness,
                       p1.eye_sharpness as a_eye_sharpness, p1.tech_sharpness as a_tech_sharpness,
                       p1.comp_score as a_comp_score, p1.power_point_score as a_power_point_score,
                       p1.leading_lines_score as a_leading_lines_score,
                       p1.exposure_score as a_exposure_score, p1.color_score as a_color_score,
                       p1.contrast_score as a_contrast_score, p1.dynamic_range_stops as a_dynamic_range_stops,
                       p1.mean_saturation as a_mean_saturation, p1.noise_sigma as a_noise_sigma,
                       p1.isolation_bonus as a_isolation_bonus,
                       p2.aesthetic as b_aesthetic, p2.quality_score as b_quality_score,
                       p2.face_quality as b_face_quality, p2.face_sharpness as b_face_sharpness,
                       p2.eye_sharpness as b_eye_sharpness, p2.tech_sharpness as b_tech_sharpness,
                       p2.comp_score as b_comp_score, p2.power_point_score as b_power_point_score,
                       p2.leading_lines_score as b_leading_lines_score,
                       p2.exposure_score as b_exposure_score, p2.color_score as b_color_score,
                       p2.contrast_score as b_contrast_score, p2.dynamic_range_stops as b_dynamic_range_stops,
                       p2.mean_saturation as b_mean_saturation, p2.noise_sigma as b_noise_sigma,
                       p2.isolation_bonus as b_isolation_bonus
                FROM comparisons c
                JOIN photos p1 ON c.photo_a_path = p1.path
                JOIN photos p2 ON c.photo_b_path = p2.path
                WHERE {where_sql}
            """, params)

            all_data = list(cursor)

            if len(all_data) < min_comparisons:
                return {
                    'error': f'Need at least {min_comparisons} comparisons (have {len(all_data)})',
                    'comparison_count': len(all_data),
                }

            if len(all_data) < n_folds:
                n_folds = len(all_data)

            # Prepare data
            n_features = len(self.SCORE_COMPONENTS)
            X_a = self._scale_features(np.array([
                [float(row[f'a_{c}'] or 0.0) for c in self.SCORE_COMPONENTS]
                for row in all_data
            ]))
            X_b = self._scale_features(np.array([
                [float(row[f'b_{c}'] or 0.0) for c in self.SCORE_COMPONENTS]
                for row in all_data
            ]))
            winners = np.array([
                1 if row['winner'] == 'a' else (-1 if row['winner'] == 'b' else 0)
                for row in all_data
            ])

            # Create fold indices
            indices = np.arange(len(all_data))
            np.random.seed(42)
            np.random.shuffle(indices)
            folds = np.array_split(indices, n_folds)

            fold_weights = []
            fold_accuracies = []

            for fold_idx in range(n_folds):
                # Test set is current fold, train set is all others
                test_indices = folds[fold_idx]
                train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != fold_idx])

                if len(train_indices) < 10:
                    continue

                # Train weights on train set
                train_X_a = X_a[train_indices]
                train_X_b = X_b[train_indices]
                train_winners = winners[train_indices]

                def neg_log_likelihood_train(weights):
                    w_sum = weights.sum()
                    if w_sum > 1e-8:
                        w = weights / w_sum
                    else:
                        w = np.ones(n_features) / n_features

                    scores_a = train_X_a @ w
                    scores_b = train_X_b @ w
                    diff = scores_a - scores_b

                    total_nll = 0.0
                    for d, winner in zip(diff, train_winners):
                        if winner == 1:
                            total_nll += np.log1p(np.exp(-np.clip(d, -20, 20)))
                        elif winner == -1:
                            total_nll += np.log1p(np.exp(np.clip(d, -20, 20)))
                        else:
                            total_nll += (d / 0.2) ** 2
                    return total_nll

                # Optimize
                bounds = [(0.0, 0.60) for _ in range(n_features)]
                constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
                start = np.ones(n_features) / n_features

                try:
                    result = minimize(
                        neg_log_likelihood_train,
                        start,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 300}
                    )
                    trained_w = np.maximum(result.x, 0.0)
                    if trained_w.sum() > 0:
                        trained_w = trained_w / trained_w.sum()
                    else:
                        trained_w = np.ones(n_features) / n_features
                except Exception:
                    trained_w = np.ones(n_features) / n_features

                fold_weights.append(trained_w)

                # Evaluate on test set
                test_X_a = X_a[test_indices]
                test_X_b = X_b[test_indices]
                test_winners = winners[test_indices]

                test_scores_a = test_X_a @ trained_w
                test_scores_b = test_X_b @ trained_w
                test_diff = test_scores_a - test_scores_b

                correct = 0
                total = 0
                for d, winner in zip(test_diff, test_winners):
                    if winner == 1 and d > 0:
                        correct += 1
                    elif winner == -1 and d < 0:
                        correct += 1
                    if winner != 0:
                        total += 1

                fold_acc = (correct / total * 100) if total > 0 else 0.0
                fold_accuracies.append(fold_acc)

            if not fold_weights:
                return {
                    'error': 'Cross-validation failed - not enough data per fold',
                    'comparison_count': len(all_data),
                }

            # Average weights across folds
            avg_weights = np.mean(fold_weights, axis=0)
            avg_weights = avg_weights / avg_weights.sum()

            # Load current weights for comparison
            old_weights = self._load_current_weights(category)

            new_weights = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, avg_weights)}

            return {
                'old_weights': old_weights,
                'new_weights': new_weights,
                'average_weights': new_weights,
                'cv_accuracy': round(np.mean(fold_accuracies), 1),
                'cv_std': round(np.std(fold_accuracies), 1),
                'fold_results': [round(a, 1) for a in fold_accuracies],
                'n_folds': len(fold_accuracies),
                'comparisons_used': len(all_data),
                'method': 'cross_validated_direct_optimization',
            }

    def compute_weight_confidence(
        self,
        category: Optional[str] = None,
        n_bootstrap: int = 100,
        min_comparisons: int = 30,
    ) -> Dict:
        """
        Bootstrap resampling to estimate weight uncertainty.

        Resamples comparisons with replacement and re-optimizes weights
        to estimate confidence intervals.

        Args:
            category: Category to analyze
            n_bootstrap: Number of bootstrap samples
            min_comparisons: Minimum comparisons required

        Returns:
            Dict with:
            - weights: point estimates
            - lower_bounds: 2.5th percentile per weight
            - upper_bounds: 97.5th percentile per weight
            - confidence_intervals: per-component CI width
            - stable_components: components with narrow CIs
        """
        with get_connection(self.db_path) as conn:
            # Get comparisons
            where_clauses = ["winner IN ('a', 'b', 'tie')"]
            params = []
            if category:
                where_clauses.append("c.category = ?")
                params.append(category)

            where_sql = " AND ".join(where_clauses)
            cursor = conn.execute(f"""
                SELECT c.photo_a_path, c.photo_b_path, c.winner,
                       p1.aesthetic as a_aesthetic, p1.quality_score as a_quality_score,
                       p1.face_quality as a_face_quality, p1.face_sharpness as a_face_sharpness,
                       p1.eye_sharpness as a_eye_sharpness, p1.tech_sharpness as a_tech_sharpness,
                       p1.comp_score as a_comp_score, p1.power_point_score as a_power_point_score,
                       p1.leading_lines_score as a_leading_lines_score,
                       p1.exposure_score as a_exposure_score, p1.color_score as a_color_score,
                       p1.contrast_score as a_contrast_score, p1.dynamic_range_stops as a_dynamic_range_stops,
                       p1.mean_saturation as a_mean_saturation, p1.noise_sigma as a_noise_sigma,
                       p1.isolation_bonus as a_isolation_bonus,
                       p2.aesthetic as b_aesthetic, p2.quality_score as b_quality_score,
                       p2.face_quality as b_face_quality, p2.face_sharpness as b_face_sharpness,
                       p2.eye_sharpness as b_eye_sharpness, p2.tech_sharpness as b_tech_sharpness,
                       p2.comp_score as b_comp_score, p2.power_point_score as b_power_point_score,
                       p2.leading_lines_score as b_leading_lines_score,
                       p2.exposure_score as b_exposure_score, p2.color_score as b_color_score,
                       p2.contrast_score as b_contrast_score, p2.dynamic_range_stops as b_dynamic_range_stops,
                       p2.mean_saturation as b_mean_saturation, p2.noise_sigma as b_noise_sigma,
                       p2.isolation_bonus as b_isolation_bonus
                FROM comparisons c
                JOIN photos p1 ON c.photo_a_path = p1.path
                JOIN photos p2 ON c.photo_b_path = p2.path
                WHERE {where_sql}
            """, params)

            all_data = list(cursor)

            if len(all_data) < min_comparisons:
                return {
                    'error': f'Need at least {min_comparisons} comparisons (have {len(all_data)})',
                    'comparison_count': len(all_data),
                }

            # Prepare data
            n_features = len(self.SCORE_COMPONENTS)
            X_a = self._scale_features(np.array([
                [float(row[f'a_{c}'] or 0.0) for c in self.SCORE_COMPONENTS]
                for row in all_data
            ]))
            X_b = self._scale_features(np.array([
                [float(row[f'b_{c}'] or 0.0) for c in self.SCORE_COMPONENTS]
                for row in all_data
            ]))
            winners = np.array([
                1 if row['winner'] == 'a' else (-1 if row['winner'] == 'b' else 0)
                for row in all_data
            ])

            bootstrap_weights = []
            np.random.seed(42)

            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(all_data), size=len(all_data), replace=True)
                boot_X_a = X_a[indices]
                boot_X_b = X_b[indices]
                boot_winners = winners[indices]

                def neg_log_likelihood_boot(weights):
                    w_sum = weights.sum()
                    if w_sum > 1e-8:
                        w = weights / w_sum
                    else:
                        w = np.ones(n_features) / n_features

                    scores_a = boot_X_a @ w
                    scores_b = boot_X_b @ w
                    diff = scores_a - scores_b

                    total_nll = 0.0
                    for d, winner in zip(diff, boot_winners):
                        if winner == 1:
                            total_nll += np.log1p(np.exp(-np.clip(d, -20, 20)))
                        elif winner == -1:
                            total_nll += np.log1p(np.exp(np.clip(d, -20, 20)))
                        else:
                            total_nll += (d / 0.2) ** 2
                    return total_nll

                bounds = [(0.0, 0.60) for _ in range(n_features)]
                constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
                start = np.ones(n_features) / n_features

                try:
                    result = minimize(
                        neg_log_likelihood_boot,
                        start,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 200}
                    )
                    boot_w = np.maximum(result.x, 0.0)
                    if boot_w.sum() > 0:
                        boot_w = boot_w / boot_w.sum()
                    else:
                        boot_w = np.ones(n_features) / n_features
                    bootstrap_weights.append(boot_w)
                except Exception:
                    continue

            if len(bootstrap_weights) < 10:
                return {
                    'error': 'Bootstrap failed - not enough successful optimizations',
                    'comparison_count': len(all_data),
                }

            bootstrap_weights = np.array(bootstrap_weights)

            # Point estimates (median)
            point_estimates = np.median(bootstrap_weights, axis=0)
            point_estimates = point_estimates / point_estimates.sum()

            # Confidence intervals (2.5th and 97.5th percentiles)
            lower_bounds = np.percentile(bootstrap_weights, 2.5, axis=0)
            upper_bounds = np.percentile(bootstrap_weights, 97.5, axis=0)

            # Identify stable components (CI width < 10%)
            ci_widths = upper_bounds - lower_bounds
            stable_threshold = 0.10
            stable_components = [
                self.SCORE_COMPONENTS[i]
                for i, width in enumerate(ci_widths)
                if width < stable_threshold
            ]

            weights = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, point_estimates)}
            lower = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, lower_bounds)}
            upper = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, upper_bounds)}
            ci_width = {c: float(w) for c, w in zip(self.SCORE_COMPONENTS, ci_widths)}

            return {
                'weights': weights,
                'lower_bounds': lower,
                'upper_bounds': upper,
                'confidence_intervals': ci_width,
                'stable_components': stable_components,
                'n_bootstrap': len(bootstrap_weights),
                'comparisons_used': len(all_data),
            }

    def _load_current_weights(self, category: Optional[str]) -> Dict[str, float]:
        """Load current weights from scoring_config.json."""
        # Mapping from DB column names to config key names
        db_to_config = {
            'aesthetic': 'aesthetic',
            'face_quality': 'face_quality',
            'eye_sharpness': 'eye_sharpness',
            'tech_sharpness': 'tech_sharpness',
            'color_score': 'color',
            'exposure_score': 'exposure',
            'comp_score': 'composition',
            'isolation_bonus': 'isolation',
            'quality_score': 'quality',
            'contrast_score': 'contrast',
            'dynamic_range_stops': 'dynamic_range',
        }

        try:
            with open(self.config_path) as f:
                config = json.load(f)

            cat = category or 'others'

            # Try v4 categories array first, then v3 flat dicts
            cat_weights = {}
            for cat_entry in config.get('categories', []):
                if cat_entry.get('name') == cat:
                    cat_weights = cat_entry.get('weights', {})
                    break
            if not cat_weights:
                if 'category_weights' in config and cat in config['category_weights']:
                    cat_weights = config['category_weights'][cat]
                elif 'weights' in config and cat in config['weights']:
                    cat_weights = config['weights'][cat]

            # Convert percent values to decimal weights
            weights = {}
            for db_col in self.SCORE_COMPONENTS:
                config_key = db_to_config.get(db_col, db_col)
                percent_key = f"{config_key}_percent"
                if percent_key in cat_weights:
                    weights[db_col] = cat_weights[percent_key] / 100.0
                else:
                    weights[db_col] = 0.0

            # If all weights are 0, use uniform distribution
            if sum(weights.values()) == 0:
                return {c: 1.0 / len(self.SCORE_COMPONENTS) for c in self.SCORE_COMPONENTS}

            return weights

        except Exception as e:
            logger.warning("Could not load weights from config: %s", e)
            # Return default uniform weights
            return {c: 1.0 / len(self.SCORE_COMPONENTS) for c in self.SCORE_COMPONENTS}

    def apply_optimized_weights(
        self,
        new_weights: Dict[str, float],
        category: str,
        backup: bool = True
    ) -> str:
        """
        Apply optimized weights to scoring_config.json.

        Args:
            new_weights: Dict of component -> weight (0.0 to 1.0)
            category: Category to update
            backup: Create backup before modifying

        Returns:
            Path to backup file (if created)
        """
        if backup:
            # Create timestamped backup
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.config_path}.backup.{timestamp}"
            shutil.copy2(self.config_path, backup_path)
        else:
            backup_path = None

        # Load current config
        with open(self.config_path) as f:
            config = json.load(f)

        # Update weights for the category — v4 categories array first, v3 fallback
        cat_weights = None
        for cat_entry in config.get('categories', []):
            if cat_entry.get('name') == category:
                if 'weights' not in cat_entry:
                    cat_entry['weights'] = {}
                cat_weights = cat_entry['weights']
                break

        if cat_weights is None:
            # v3 fallback
            if 'weights' not in config:
                config['weights'] = {}
            if category not in config['weights']:
                config['weights'][category] = {}
            cat_weights = config['weights'][category]

        # Mapping from DB column names to config key names
        db_to_config = {
            'aesthetic': 'aesthetic',
            'face_quality': 'face_quality',
            'eye_sharpness': 'eye_sharpness',
            'tech_sharpness': 'tech_sharpness',
            'color_score': 'color',
            'exposure_score': 'exposure',
            'comp_score': 'composition',
            'isolation_bonus': 'isolation',
            'quality_score': 'quality',
            'contrast_score': 'contrast',
            'dynamic_range_stops': 'dynamic_range',
        }

        # Convert decimal weights to percentages
        for component, weight in new_weights.items():
            config_key = db_to_config.get(component, component)
            key = f"{config_key}_percent"
            cat_weights[key] = round(weight * 100, 1)

        # Post-rounding normalization to ensure weights sum to exactly 100%
        percent_keys = [f"{db_to_config.get(c, c)}_percent" for c in new_weights.keys()]
        total = sum(cat_weights[k] for k in percent_keys if k in cat_weights)
        if total > 0 and abs(total - 100.0) > 0.01:
            # Adjust largest weight to compensate for rounding error
            adjustment = 100.0 - total
            largest_key = max(percent_keys, key=lambda k: cat_weights.get(k, 0))
            cat_weights[largest_key] = round(cat_weights[largest_key] + adjustment, 1)
            logger.info("Adjusted %s by %+.1f%% to ensure 100%% total", largest_key, adjustment)

        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return backup_path

    def get_optimization_history(self, limit: int = 10) -> List[Dict]:
        """Get recent optimization runs."""
        with get_connection(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM weight_optimization_runs
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor]

def print_comparison_stats(db_path: str = DEFAULT_DB_PATH):
    """Print comparison statistics for CLI."""
    from comparison import ComparisonManager

    manager = ComparisonManager(db_path)
    stats = manager.get_statistics()

    logger.info("=" * 60)
    logger.info("PAIRWISE COMPARISON STATISTICS")
    logger.info("=" * 60)

    logger.info("Total comparisons: %d", stats['total_comparisons'])
    logger.info("Unique photos compared: %d", stats['unique_photos_compared'])

    logger.info("Winner breakdown:")
    for winner, count in stats['winner_breakdown'].items():
        logger.info("  %s: %d", winner, count)

    if stats['category_breakdown']:
        logger.info("By category:")
        for cat_stat in stats['category_breakdown'][:5]:
            logger.info("  %s: %d", cat_stat['category'], cat_stat['count'])

    if stats['recent_optimization_runs']:
        logger.info("Recent optimization runs:")
        for run in stats['recent_optimization_runs']:
            before = run.get('mse_before') or 0.0
            after = run.get('mse_after') or 0.0
            improvement = after - before
            logger.info(
                "  %s: accuracy %.1f%% -> %.1f%% (%+.1f pp)",
                run['timestamp'][:10],
                before,
                after,
                improvement,
            )

    logger.info("=" * 60)


def run_weight_optimization(
    db_path: str = DEFAULT_DB_PATH,
    config_path: str = 'scoring_config.json',
    min_comparisons: int = 30,
):
    """Run weight optimization from CLI. Optimizes and saves weights automatically."""
    optimizer = WeightOptimizer(db_path, config_path)

    logger.info("=" * 60)
    logger.info("WEIGHT OPTIMIZATION")
    logger.info("=" * 60)

    logger.info("Optimizing weights via direct preference optimization...")
    result = optimizer.optimize_weights_direct(
        category=None,
        min_comparisons=min_comparisons,
    )

    if 'error' in result:
        logger.error("Error: %s", result['error'])
        return

    logger.info("Comparisons used: %d", result['comparisons_used'])
    logger.info("Accuracy before: %.1f%%", result['accuracy_before'])
    logger.info("Accuracy after:  %.1f%%", result['accuracy_after'])
    logger.info("Improvement:     %+.1f pp", result['improvement'])
    logger.info("Log-likelihood:  %.4f", result.get('log_likelihood', 0.0))

    logger.info("Optimized weights:")
    for component, weight in sorted(result['new_weights'].items(), key=lambda x: -x[1]):
        if weight > 0.01:
            logger.info("  %s: %.1f%%", component, weight * 100)

    logger.info("Applying weights to config...")
    backup_path = optimizer.apply_optimized_weights(
        result['new_weights'],
        category='others',
    )
    if backup_path:
        logger.info("  Backup created: %s", backup_path)
    logger.info("  Config updated: %s", config_path)
    logger.info("Run 'python facet.py --recalculate' to apply new weights to scores.")

    logger.info("=" * 60)
