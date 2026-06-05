import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { TranslatePipe } from './translate.pipe';
import { I18nService } from '../../core/services/i18n.service';

describe('TranslatePipe', () => {
  let pipe: TranslatePipe;
  let i18nMock: { t: Mock; translations: Mock };
  let currentTranslations: Record<string, unknown>;

  beforeEach(() => {
    currentTranslations = { en: true };
    i18nMock = {
      t: vi.fn(),
      translations: vi.fn(() => currentTranslations),
    };

    TestBed.configureTestingModule({
      providers: [
        TranslatePipe,
        { provide: I18nService, useValue: i18nMock },
      ],
    });

    pipe = TestBed.inject(TranslatePipe);
  });

  it('delegates to I18nService.t()', () => {
    i18nMock.t.mockReturnValue('Hello');

    const result = pipe.transform('greeting');

    expect(i18nMock.t).toHaveBeenCalledWith('greeting', undefined);
    expect(result).toBe('Hello');
  });

  it('passes vars through to I18nService.t()', () => {
    i18nMock.t.mockReturnValue('Hello Alice');

    const vars = { name: 'Alice', count: 3 };
    const result = pipe.transform('greeting', vars);

    expect(i18nMock.t).toHaveBeenCalledWith('greeting', vars);
    expect(result).toBe('Hello Alice');
  });

  it('returns key when I18nService returns key (no translation)', () => {
    i18nMock.t.mockReturnValue('missing.key');

    const result = pipe.transform('missing.key');

    expect(i18nMock.t).toHaveBeenCalledWith('missing.key', undefined);
    expect(result).toBe('missing.key');
  });

  it('memoises repeated transforms for the same key+vars+translations', () => {
    i18nMock.t.mockReturnValue('Hello');

    pipe.transform('greeting');
    pipe.transform('greeting');
    pipe.transform('greeting');

    expect(i18nMock.t).toHaveBeenCalledTimes(1);
  });

  it('recomputes when the translations object reference changes', () => {
    i18nMock.t.mockReturnValueOnce('Hello').mockReturnValueOnce('Bonjour');

    expect(pipe.transform('greeting')).toBe('Hello');

    currentTranslations = { fr: true };

    expect(pipe.transform('greeting')).toBe('Bonjour');
    expect(i18nMock.t).toHaveBeenCalledTimes(2);
  });

  it('recomputes when the vars change', () => {
    i18nMock.t
      .mockReturnValueOnce('Hello Alice')
      .mockReturnValueOnce('Hello Bob');

    expect(pipe.transform('greeting', { name: 'Alice' })).toBe('Hello Alice');
    expect(pipe.transform('greeting', { name: 'Bob' })).toBe('Hello Bob');
    expect(i18nMock.t).toHaveBeenCalledTimes(2);
  });
});
