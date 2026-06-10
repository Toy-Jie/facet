import { TestBed } from '@angular/core/testing';
import { MatSnackBar } from '@angular/material/snack-bar';
import { Subject } from 'rxjs';
import { UndoService } from './undo.service';
import { I18nService } from './i18n.service';

interface FakeSnackRef {
  onAction: Subject<void>;
  afterDismissed: Subject<void>;
}

describe('UndoService', () => {
  let service: UndoService;
  let refs: FakeSnackRef[];
  let snackOpen: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    refs = [];
    snackOpen = vi.fn(() => {
      const ref: FakeSnackRef = { onAction: new Subject(), afterDismissed: new Subject() };
      refs.push(ref);
      return {
        onAction: () => ref.onAction.asObservable(),
        afterDismissed: () => ref.afterDismissed.asObservable(),
      };
    });
    TestBed.configureTestingModule({
      providers: [
        UndoService,
        { provide: MatSnackBar, useValue: { open: snackOpen } },
        { provide: I18nService, useValue: { t: (k: string) => k } },
      ],
    });
    service = TestBed.inject(UndoService);
  });

  it('invert command: undo action runs undo()', async () => {
    const undo = vi.fn(() => Promise.resolve());
    service.register({ labelKey: 'x', undo });
    refs[0].onAction.next();
    await Promise.resolve();
    expect(undo).toHaveBeenCalled();
  });

  it('deferred command: commit fires on dismissal timeout', async () => {
    const commit = vi.fn(() => Promise.resolve());
    const undo = vi.fn(() => Promise.resolve());
    service.register({ labelKey: 'x', commit, undo });
    refs[0].afterDismissed.next();
    await Promise.resolve();
    expect(commit).toHaveBeenCalled();
    expect(undo).not.toHaveBeenCalled();
  });

  it('deferred command: undo cancels the commit', async () => {
    const commit = vi.fn(() => Promise.resolve());
    const undo = vi.fn(() => Promise.resolve());
    service.register({ labelKey: 'x', commit, undo });
    refs[0].onAction.next();
    refs[0].afterDismissed.next();
    await Promise.resolve();
    expect(undo).toHaveBeenCalled();
    expect(commit).not.toHaveBeenCalled();
  });

  it('registering a second command commits a pending deferred one first', async () => {
    const firstCommit = vi.fn(() => Promise.resolve());
    service.register({ labelKey: 'a', commit: firstCommit, undo: () => Promise.resolve() });
    service.register({ labelKey: 'b', undo: () => Promise.resolve() });
    await Promise.resolve();
    expect(firstCommit).toHaveBeenCalled();
  });

  it('flushPending commits and clears the pending command', async () => {
    const commit = vi.fn(() => Promise.resolve());
    service.register({ labelKey: 'x', commit, undo: () => Promise.resolve() });
    await service.flushPending();
    expect(commit).toHaveBeenCalledTimes(1);
    // Dismissal after flush must not double-commit
    refs[0].afterDismissed.next();
    await Promise.resolve();
    expect(commit).toHaveBeenCalledTimes(1);
  });

  it('shows confirmation snackbar after successful undo', async () => {
    service.register({ labelKey: 'x', undo: () => Promise.resolve() });
    refs[0].onAction.next();
    await Promise.resolve();
    await Promise.resolve();
    expect(snackOpen).toHaveBeenCalledWith('undo.restored', '', expect.anything());
  });
});
