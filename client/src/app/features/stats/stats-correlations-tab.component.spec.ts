import { TestBed } from '@angular/core/testing';
import { signal } from '@angular/core';
import { of } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { StatsFiltersService } from './stats-filters.service';
import { StatsCorrelationsTabComponent } from './stats-correlations-tab.component';
import { downloadCsv } from '../../shared/utils/csv';

jest.mock('../../shared/utils/csv', () => ({ downloadCsv: jest.fn() }));

describe('StatsCorrelationsTabComponent', () => {
  let component: StatsCorrelationsTabComponent;

  beforeEach(() => {
    jest.mocked(downloadCsv).mockClear();
    TestBed.configureTestingModule({
      providers: [
        StatsCorrelationsTabComponent,
        { provide: ApiService, useValue: { get: jest.fn(() => of({})) } },
        { provide: I18nService, useValue: { t: jest.fn((k: string) => k), currentLang: jest.fn(() => 'en') } },
        { provide: StatsFiltersService, useValue: { filterCategory: signal(''), dateFrom: signal(''), dateTo: signal('') } },
      ],
    });
    component = TestBed.inject(StatsCorrelationsTabComponent);
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('exportCsv is a no-op when no correlation data is loaded', () => {
    component.corrData.set(null);
    component.exportCsv();
    expect(jest.mocked(downloadCsv)).not.toHaveBeenCalled();
  });

  it('exportCsv builds one row per label from the metrics branch', () => {
    component.corrData.set({
      labels: ['2023', '2024'],
      metrics: { aggregate: [7.1, 7.8], aesthetic: [6.0, 6.5] },
      x_axis: 'date_year',
      group_by: '',
    });
    component.exportCsv();
    const [filename, records] = jest.mocked(downloadCsv).mock.calls[0];
    expect(filename).toBe('facet-correlations');
    expect(records).toEqual([
      { date_year: '2023', aggregate: 7.1, aesthetic: 6.0 },
      { date_year: '2024', aggregate: 7.8, aesthetic: 6.5 },
    ]);
  });

  it('exportCsv builds one row per group+label from the groups branch', () => {
    component.corrYMetrics.set(['aggregate']);
    component.corrData.set({
      labels: ['2023', '2024'],
      groups: {
        canon: { '2023': { aggregate: 7.0 }, '2024': { aggregate: 7.5 } },
        nikon: { '2023': { aggregate: 6.8 }, '2024': { aggregate: 7.2 } },
      },
      x_axis: 'date_year',
      group_by: 'camera_model',
    });
    component.exportCsv();
    const [, records] = jest.mocked(downloadCsv).mock.calls[0];
    expect(records).toEqual([
      { date_year: '2023', group: 'canon', aggregate: 7.0 },
      { date_year: '2024', group: 'canon', aggregate: 7.5 },
      { date_year: '2023', group: 'nikon', aggregate: 6.8 },
      { date_year: '2024', group: 'nikon', aggregate: 7.2 },
    ]);
  });
});
