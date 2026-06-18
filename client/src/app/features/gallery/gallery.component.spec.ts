import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { of } from 'rxjs';
import { MatDialog } from '@angular/material/dialog';
import { MatSnackBar } from '@angular/material/snack-bar';
import { GalleryStore, GalleryFilters, DEFAULT_FILTERS } from './gallery.store';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { I18nService } from '../../core/services/i18n.service';
import { AlbumService } from '../../core/services/album.service';
import { GalleryComponent } from './gallery.component';
import { ScoreClassPipe } from '../../shared/pipes/score.pipes';

describe('GalleryComponent', () => {
  let component: GalleryComponent;

   
  let mockStore: any;
  let mockApi: { thumbnailUrl: Mock; downloadUrl: Mock; getRaw: Mock };
  let mockAuth: { isEdition: Mock };
  let mockI18n: { t: Mock };

  beforeEach(() => {
    mockStore = {
      filters: signal<GalleryFilters>({ ...DEFAULT_FILTERS }),
      types: signal([
        { id: 'portrait', label: 'Portrait', count: 100 },
        { id: 'landscape', label: 'Landscape', count: 200 },
        { id: 'macro', label: 'Macro', count: 50 },
      ]),
      photos: signal([]),
      total: signal(0),
      loading: signal(false),
      hasMore: signal(false),
      scanDirectories: signal([]),
      scanDirectoriesLoading: signal(false),
      cameras: signal([]),
      lenses: signal([]),
      tags: signal([]),
      persons: signal([]),
      config: signal(null),
      activeFilterCount: signal(0),
      filterDrawerOpen: signal(false),
      currentAlbum: signal(null),
      initializing: signal(false),
      galleryMode: signal('mosaic'),
      cardWidth: signal(300),
      setFilterDrawerOpen: vi.fn(),
      loadConfig: vi.fn(() => Promise.resolve()),
      loadFilterOptions: vi.fn(() => Promise.resolve()),
      loadTypeCounts: vi.fn(() => Promise.resolve()),
      loadScanDirectories: vi.fn(() => Promise.resolve()),
      loadPhotos: vi.fn(() => Promise.resolve()),
      updateFilter: vi.fn(() => Promise.resolve()),
      resetFilters: vi.fn(() => Promise.resolve()),
      nextPage: vi.fn(() => Promise.resolve()),
      toggleFavorite: vi.fn(),
      toggleRejected: vi.fn(),
      selectedPaths: signal(new Set<string>()),
      selectionCount: signal(0),
      toggleSelection: vi.fn(),
      selectAllLoaded: vi.fn(),
      clearSelection: vi.fn(),
      restoreSelection: vi.fn(),
      restoreSnapshot: vi.fn(() => Promise.resolve()),
      viewSnapshot: signal(null),
      filterKey: vi.fn(() => '{}'),
      hiddenSummary: signal({ total: 0, blinks: 0, bursts: 0, duplicates: 0 }),
      updateFilters: vi.fn(() => Promise.resolve()),
      selectScanDirectory: vi.fn(() => Promise.resolve()),
      clearScanDirectory: vi.fn(() => Promise.resolve()),
      setRating: vi.fn(),
      batchFavorite: vi.fn(() => Promise.resolve(new Map())),
      batchReject: vi.fn(() => Promise.resolve(new Map())),
      batchRating: vi.fn(() => Promise.resolve(new Map())),
    };

    mockApi = {
      thumbnailUrl: vi.fn((path: string) => `/thumbnail?path=${path}`),
      downloadUrl: vi.fn((path: string) => `/download?path=${path}`),
      getRaw: vi.fn(() => of(new Blob(['photo']))),
    };

    mockAuth = { isEdition: vi.fn(() => false) };

    mockI18n = {
      t: vi.fn((key: string) => key),
    };

    TestBed.configureTestingModule({
      providers: [
        { provide: GalleryStore, useValue: mockStore },
        { provide: ApiService, useValue: mockApi },
        { provide: AuthService, useValue: mockAuth },
        { provide: I18nService, useValue: mockI18n },
        { provide: AlbumService, useValue: { list: vi.fn(() => of({ albums: [] })), get: vi.fn(() => of({})) } },
        { provide: ActivatedRoute, useValue: { snapshot: { paramMap: { get: vi.fn(() => null) } } } },
        { provide: MatDialog, useValue: { open: vi.fn() } },
        { provide: MatSnackBar, useValue: { open: vi.fn() } },
      ],
    });
    component = TestBed.runInInjectionContext(() => new GalleryComponent());
  });

  describe('ScoreClassPipe', () => {
    let pipe: ScoreClassPipe;

    beforeEach(() => {
      pipe = new ScoreClassPipe();
    });

    it('should return green class for score >= 8 (no config)', () => {
      expect(pipe.transform(8, null)).toBe('bg-green-600 text-white');
      expect(pipe.transform(9.5, null)).toBe('bg-green-600 text-white');
      expect(pipe.transform(10, null)).toBe('bg-green-600 text-white');
    });

    it('should return yellow class for score >= 6 and < 8 (no config)', () => {
      expect(pipe.transform(6, null)).toBe('bg-yellow-600 text-white');
      expect(pipe.transform(7.9, null)).toBe('bg-yellow-600 text-white');
    });

    it('should return orange class for score >= 4 and < 6 (no config)', () => {
      expect(pipe.transform(4, null)).toBe('bg-orange-600 text-white');
      expect(pipe.transform(5.9, null)).toBe('bg-orange-600 text-white');
    });

    it('should return red class for score < 4 (no config)', () => {
      expect(pipe.transform(3.9, null)).toBe('bg-red-600 text-white');
      expect(pipe.transform(0, null)).toBe('bg-red-600 text-white');
      expect(pipe.transform(1, null)).toBe('bg-red-600 text-white');
    });

    it('should use config thresholds when provided', () => {
      const config = { quality_thresholds: { excellent: 9, great: 7, good: 5, best: 10 } };
      expect(pipe.transform(9, config)).toBe('bg-green-600 text-white');
      expect(pipe.transform(7, config)).toBe('bg-yellow-600 text-white');
      expect(pipe.transform(5, config)).toBe('bg-orange-600 text-white');
      expect(pipe.transform(4, config)).toBe('bg-red-600 text-white');
    });
  });

  describe('ngOnInit()', () => {
    it('should call store.loadConfig, loadFilterOptions, loadTypeCounts, and loadScanDirectories', async () => {
      await component.ngOnInit();

      expect(mockStore.loadConfig).toHaveBeenCalled();
      expect(mockStore.loadFilterOptions).toHaveBeenCalled();
      expect(mockStore.loadTypeCounts).toHaveBeenCalled();
      expect(mockStore.loadScanDirectories).toHaveBeenCalled();
      expect(mockStore.loadPhotos).not.toHaveBeenCalled();
    });

    it('should call loadConfig before loadFilterOptions and loadTypeCounts', async () => {
      const callOrder: string[] = [];
      mockStore.loadConfig.mockImplementation(() => {
        callOrder.push('loadConfig');
        return Promise.resolve();
      });
      mockStore.loadFilterOptions.mockImplementation(() => {
        callOrder.push('loadFilterOptions');
        return Promise.resolve();
      });
      mockStore.loadTypeCounts.mockImplementation(() => {
        callOrder.push('loadTypeCounts');
        return Promise.resolve();
      });
      mockStore.loadPhotos.mockImplementation(() => {
        callOrder.push('loadPhotos');
        return Promise.resolve();
      });

      await component.ngOnInit();

      expect(callOrder.indexOf('loadConfig')).toBeLessThan(
        callOrder.indexOf('loadFilterOptions'),
      );
      expect(callOrder.indexOf('loadConfig')).toBeLessThan(
        callOrder.indexOf('loadTypeCounts'),
      );
    });

    it('should call loadPhotos after loadFilterOptions and loadTypeCounts when a scan directory is selected', async () => {
      const callOrder: string[] = [];
      mockStore.filters.set({ ...DEFAULT_FILTERS, path_prefix: '/photos/shoot-a' });
      mockStore.loadConfig.mockImplementation(() => {
        callOrder.push('loadConfig');
        return Promise.resolve();
      });
      mockStore.loadFilterOptions.mockImplementation(() => {
        callOrder.push('loadFilterOptions');
        return Promise.resolve();
      });
      mockStore.loadTypeCounts.mockImplementation(() => {
        callOrder.push('loadTypeCounts');
        return Promise.resolve();
      });
      mockStore.loadScanDirectories.mockImplementation(() => {
        callOrder.push('loadScanDirectories');
        return Promise.resolve();
      });
      mockStore.loadPhotos.mockImplementation(() => {
        callOrder.push('loadPhotos');
        return Promise.resolve();
      });

      await component.ngOnInit();

      expect(callOrder.indexOf('loadPhotos')).toBeGreaterThan(
        callOrder.indexOf('loadFilterOptions'),
      );
      expect(callOrder.indexOf('loadPhotos')).toBeGreaterThan(
        callOrder.indexOf('loadTypeCounts'),
      );
      expect(callOrder.indexOf('loadPhotos')).toBeGreaterThan(
        callOrder.indexOf('loadScanDirectories'),
      );
    });
  });

  describe('selection shortcuts', () => {
    const shortcutEvent = (key: string, extra: Partial<KeyboardEvent> = {}) => ({
      key,
      target: null,
      preventDefault: vi.fn(),
      altKey: false,
      ctrlKey: false,
      metaKey: false,
      ...extra,
    }) as unknown as KeyboardEvent & { preventDefault: Mock };

    beforeEach(() => {
      mockStore.photos.set([{ path: '/one.jpg' }, { path: '/two.jpg' }]);
      mockStore.selectedPaths.set(new Set(['/one.jpg']));
      mockStore.selectionCount.set(1);
    });

    it('selects all loaded photos with Ctrl+A', () => {
      const event = shortcutEvent('a', { ctrlKey: true });

      (component as any).onSelectionShortcut(event);

      expect(event.preventDefault).toHaveBeenCalled();
      expect(mockStore.selectAllLoaded).toHaveBeenCalled();
    });

    it('clears the current selection with Escape', () => {
      const event = shortcutEvent('Escape');

      (component as any).onSelectionShortcut(event);

      expect(event.preventDefault).toHaveBeenCalled();
      expect(mockStore.clearSelection).toHaveBeenCalled();
    });

    it('favorites selected photos with F in edition mode', () => {
      mockAuth.isEdition.mockReturnValue(true);
      const batchFavorite = vi.spyOn(component as any, 'batchFavorite').mockResolvedValue(undefined);
      const event = shortcutEvent('f');

      (component as any).onSelectionShortcut(event);

      expect(event.preventDefault).toHaveBeenCalled();
      expect(batchFavorite).toHaveBeenCalled();
    });

    it('rejects selected photos with X in edition mode', () => {
      mockAuth.isEdition.mockReturnValue(true);
      const batchReject = vi.spyOn(component as any, 'batchReject').mockResolvedValue(undefined);
      const event = shortcutEvent('x');

      (component as any).onSelectionShortcut(event);

      expect(event.preventDefault).toHaveBeenCalled();
      expect(batchReject).toHaveBeenCalled();
    });

    it('downloads selected photos with D', () => {
      const downloadSelected = vi.spyOn(component as any, 'downloadSelected').mockResolvedValue(undefined);
      const event = shortcutEvent('d');

      (component as any).onSelectionShortcut(event);

      expect(event.preventDefault).toHaveBeenCalled();
      expect(downloadSelected).toHaveBeenCalled();
    });
  });
});
