import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatTooltipModule } from '@angular/material/tooltip';
import { AuthService } from '../../core/services/auth.service';
import { I18nService } from '../../core/services/i18n.service';
import { ScanDirectory, ScanService } from '../../core/services/scan.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';

@Component({
  selector: 'app-scan',
  standalone: true,
  imports: [
    MatButtonModule,
    MatCheckboxModule,
    MatIconModule,
    MatProgressBarModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatTooltipModule,
    TranslatePipe,
  ],
  host: { class: 'block px-4 pt-4 pb-8' },
  template: `
    <div class="mx-auto w-full max-w-5xl flex flex-col gap-4">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 class="text-2xl font-semibold">{{ 'scan.title' | translate }}</h1>
          <p class="text-sm opacity-70 mt-1">{{ 'scan.description' | translate }}</p>
        </div>
        <div class="flex items-center gap-2">
          <span class="inline-flex items-center gap-1 text-sm px-2 py-1 rounded border border-[var(--mat-sys-outline-variant)]">
            <mat-icon class="!text-base !w-4 !h-4">{{ status().running ? 'sync' : statusIcon() }}</mat-icon>
            {{ statusLabelKey() | translate }}
          </span>
          <button mat-button (click)="refresh()" [disabled]="loading()">
            <mat-icon>refresh</mat-icon>
            {{ 'scan.refresh' | translate }}
          </button>
        </div>
      </div>

      @if (!canScan()) {
        <div class="rounded-md border border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface-container)] px-4 py-6 text-sm">
          {{ 'scan.superadmin_required' | translate }}
        </div>
      } @else {
        <section class="rounded-md border border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface-container)]">
          <div class="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-[var(--mat-sys-outline-variant)]">
            <div class="font-medium">{{ 'scan.directories' | translate }}</div>
            <div class="flex items-center gap-1">
              <button mat-button (click)="selectAll()" [disabled]="!directories().length || status().running">
                <mat-icon>select_all</mat-icon>
                {{ 'scan.select_all' | translate }}
              </button>
              <button mat-button (click)="clearSelection()" [disabled]="!selectedCount() || status().running">
                <mat-icon>close</mat-icon>
                {{ 'scan.clear' | translate }}
              </button>
            </div>
          </div>

          @if (loading()) {
            <div class="flex justify-center py-10">
              <mat-spinner diameter="40" />
            </div>
          } @else if (loadError()) {
            <div class="px-4 py-8 text-sm text-red-400">{{ 'scan.load_error' | translate }}</div>
          } @else if (!directories().length) {
            <div class="px-4 py-8 text-sm opacity-70">{{ 'scan.no_directories' | translate }}</div>
          } @else {
            <div class="divide-y divide-[var(--mat-sys-outline-variant)]">
              @for (directory of directories(); track directory.path) {
                <label class="flex items-center gap-3 px-4 py-3 hover:bg-[var(--mat-sys-surface-container-high)] cursor-pointer">
                  <mat-checkbox
                    [checked]="selectedPaths().has(directory.path)"
                    [disabled]="status().running"
                    (change)="toggleDirectory(directory.path, $event.checked)" />
                  <mat-icon class="opacity-70">folder</mat-icon>
                  <div class="min-w-0 flex-1">
                    <div class="text-sm font-medium truncate" [matTooltip]="directory.path">{{ directory.path }}</div>
                    <div class="text-xs opacity-60">{{ ownerLabel(directory) }}</div>
                  </div>
                </label>
              }
            </div>
          }
        </section>

        <section class="rounded-md border border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface-container)] px-4 py-4">
          <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
            <div class="text-sm opacity-80">
              {{ 'scan.selected_count' | translate:{ count: selectedCount() } }}
            </div>
            <button mat-flat-button (click)="startScan()" [disabled]="!selectedCount() || status().running || starting()">
              @if (starting()) {
                <mat-spinner diameter="18" class="!inline-block !align-baseline" />
              } @else {
                <mat-icon>play_arrow</mat-icon>
              }
              {{ starting() ? ('scan.starting' | translate) : ('scan.start_button' | translate) }}
            </button>
          </div>

          @if (status().running || status().progress || status().exit_code !== null) {
            <div class="flex flex-col gap-3">
              @if (progressPercent() !== null) {
                <mat-progress-bar mode="determinate" [value]="progressPercent()!" />
              } @else if (status().running) {
                <mat-progress-bar mode="indeterminate" />
              }

              <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
                <div>
                  <div class="opacity-60">{{ 'scan.phase' | translate }}</div>
                  <div class="font-medium">{{ status().progress?.phase || '-' }}</div>
                </div>
                <div>
                  <div class="opacity-60">{{ 'scan.progress' | translate }}</div>
                  <div class="font-medium">{{ progressText() }}</div>
                </div>
                <div>
                  <div class="opacity-60">{{ 'scan.elapsed' | translate }}</div>
                  <div class="font-medium">{{ elapsedText() }}</div>
                </div>
              </div>

              @if (status().progress?.current_file) {
                <div class="text-xs opacity-70 truncate" [matTooltip]="status().progress?.current_file || ''">
                  {{ 'scan.current_file' | translate }}: {{ status().progress?.current_file }}
                </div>
              }
            </div>
          }
        </section>

        <section class="rounded-md border border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface-container)]">
          <div class="px-4 py-3 border-b border-[var(--mat-sys-outline-variant)] font-medium">
            {{ 'scan.output' | translate }}
          </div>
          <pre class="m-0 p-4 max-h-[360px] overflow-auto text-xs leading-5 whitespace-pre-wrap font-mono bg-black/20">{{ outputText() || ('scan.no_output' | translate) }}</pre>
        </section>
      }
    </div>
  `,
})
export class ScanComponent implements OnInit {
  protected readonly scan = inject(ScanService);
  protected readonly auth = inject(AuthService);
  private readonly i18n = inject(I18nService);
  private readonly snackBar = inject(MatSnackBar);

  protected readonly directories = signal<ScanDirectory[]>([]);
  protected readonly selectedPaths = signal<Set<string>>(new Set());
  protected readonly loading = signal(false);
  protected readonly starting = signal(false);
  protected readonly loadError = signal(false);
  protected readonly status = this.scan.status;

  protected readonly selectedCount = computed(() => this.selectedPaths().size);

  protected readonly progressPercent = computed(() => {
    const progress = this.status().progress;
    if (!progress?.total || progress.total <= 0 || progress.current === undefined) return null;
    return Math.max(0, Math.min(100, Math.round((progress.current / progress.total) * 100)));
  });

  protected readonly statusLabelKey = computed(() => {
    const status = this.status();
    if (status.running) return 'scan.running';
    if (status.exit_code === null) return 'scan.idle';
    return status.exit_code === 0 ? 'scan.completed' : 'scan.failed';
  });

  protected readonly statusIcon = computed(() => {
    const status = this.status();
    if (status.exit_code === null) return 'radio_button_unchecked';
    return status.exit_code === 0 ? 'check_circle' : 'error';
  });

  protected readonly outputText = computed(() => this.status().output.join('\n'));
  protected readonly canScan = computed(() =>
    !!this.auth.features()['show_scan_button']
    && (this.auth.isMultiUser() ? this.auth.isSuperadmin() : this.auth.isEdition()),
  );

  async ngOnInit(): Promise<void> {
    this.scan.connect();
    await this.refresh();
  }

  protected async refresh(): Promise<void> {
    if (!this.canScan()) return;
    this.loading.set(true);
    this.loadError.set(false);
    try {
      const directories = await this.scan.loadDirectories();
      this.directories.set(directories);
      const valid = new Set(directories.map(d => d.path));
      this.selectedPaths.update(paths => new Set([...paths].filter(path => valid.has(path))));
    } catch {
      this.loadError.set(true);
    } finally {
      this.loading.set(false);
    }
  }

  protected toggleDirectory(path: string, checked: boolean): void {
    this.selectedPaths.update(paths => {
      const next = new Set(paths);
      if (checked) next.add(path);
      else next.delete(path);
      return next;
    });
  }

  protected selectAll(): void {
    this.selectedPaths.set(new Set(this.directories().map(d => d.path)));
  }

  protected clearSelection(): void {
    this.selectedPaths.set(new Set());
  }

  protected async startScan(): Promise<void> {
    const directories = [...this.selectedPaths()];
    if (!directories.length || this.status().running) return;
    this.starting.set(true);
    try {
      await this.scan.startScan(directories);
      this.snackBar.open(this.i18n.t('scan.started'), '', { duration: 2000 });
    } catch {
      this.snackBar.open(this.i18n.t('scan.start_error'), '', { duration: 3000 });
    } finally {
      this.starting.set(false);
    }
  }

  protected ownerLabel(directory: ScanDirectory): string {
    return directory.owner === 'shared'
      ? this.i18n.t('scan.owner_shared')
      : this.i18n.t('scan.owner_user', { owner: directory.owner });
  }

  protected progressText(): string {
    const progress = this.status().progress;
    if (!progress) return '-';
    if (progress.total && progress.current !== undefined) {
      return `${progress.current} / ${progress.total}`;
    }
    return progress.phase || '-';
  }

  protected elapsedText(): string {
    const elapsed = this.status().elapsed_seconds;
    if (elapsed === null || elapsed === undefined) return '-';
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
  }
}
