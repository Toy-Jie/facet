import type { Mock, MockedFunction } from 'vitest';
import { TestBed } from '@angular/core/testing';
import {
  HttpBackend,
  HttpRequest,
  HttpHandlerFn,
  HttpErrorResponse,
} from '@angular/common/http';
import { EMPTY, throwError } from 'rxjs';
import { MatSnackBar } from '@angular/material/snack-bar';
import { errorInterceptor } from './error.interceptor';
import { AuthService } from '../services/auth.service';
import { I18nService } from '../services/i18n.service';

describe('errorInterceptor', () => {
  let authMock: { token: string | null; logout: Mock };
  let snackBarMock: { open: Mock };
  let i18nMock: { t: Mock; locale: Mock };
  let backendMock: { handle: Mock };
  let next: MockedFunction<HttpHandlerFn>;

  beforeEach(() => {
    authMock = { token: null, logout: vi.fn() };
    snackBarMock = { open: vi.fn() };
    i18nMock = { t: vi.fn((key: string) => key), locale: vi.fn(() => 'en') };
    // The interceptor uses HttpBackend directly to post 5xx crash reports
    // without recursing through itself. Mock returns an empty observable so
    // the post is a no-op.
    backendMock = { handle: vi.fn(() => EMPTY) };
    next = vi.fn();

    TestBed.configureTestingModule({
      providers: [
        { provide: AuthService, useValue: authMock },
        { provide: MatSnackBar, useValue: snackBarMock },
        { provide: I18nService, useValue: i18nMock },
        { provide: HttpBackend, useValue: backendMock },
      ],
    });
  });

  const runInterceptor = (req: HttpRequest<unknown>) =>
    TestBed.runInInjectionContext(() => errorInterceptor(req, next));

  it('calls auth.logout() on 401 for non-auth URLs', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error = new HttpErrorResponse({ status: 401, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        error: () => {
          expect(authMock.logout).toHaveBeenCalled();
          resolve();
        },
      });
    }));

  it('does NOT call auth.logout() on 401 for /api/auth/ URLs', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/auth/status');
      const error = new HttpErrorResponse({ status: 401, url: '/api/auth/status' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        error: () => {
          expect(authMock.logout).not.toHaveBeenCalled();
          resolve();
        },
      });
    }));

  it('does NOT call auth.logout() on other error codes (404, 500)', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error404 = new HttpErrorResponse({ status: 404, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error404));

      runInterceptor(req).subscribe({
        error: () => {
          expect(authMock.logout).not.toHaveBeenCalled();

          const error500 = new HttpErrorResponse({ status: 500, url: '/api/photos' });
          next.mockReturnValue(throwError(() => error500));

          runInterceptor(req).subscribe({
            error: () => {
              expect(authMock.logout).not.toHaveBeenCalled();
              resolve();
            },
          });
        },
      });
    }));

  it('re-throws the error', () =>
    new Promise<void>((resolve, reject) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error = new HttpErrorResponse({ status: 401, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        next: () => {
          reject(new Error('expected an error'));
        },
        error: (err: HttpErrorResponse) => {
          expect(err.status).toBe(401);
          resolve();
        },
      });
    }));

  it('shows snackbar on 429 rate limit', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error = new HttpErrorResponse({ status: 429, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        error: () => {
          expect(snackBarMock.open).toHaveBeenCalledWith('errors.rate_limited', '', { duration: 5000 });
          resolve();
        },
      });
    }));

  it('shows snackbar on 403 for non-auth URLs', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error = new HttpErrorResponse({ status: 403, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        error: () => {
          expect(snackBarMock.open).toHaveBeenCalledWith('errors.access_denied', '', { duration: 3000 });
          resolve();
        },
      });
    }));

  it('shows snackbar on 500 server error', () =>
    new Promise<void>((resolve) => {
      const req = new HttpRequest('GET', '/api/photos');
      const error = new HttpErrorResponse({ status: 500, url: '/api/photos' });
      next.mockReturnValue(throwError(() => error));

      runInterceptor(req).subscribe({
        error: () => {
          expect(snackBarMock.open).toHaveBeenCalledWith('errors.server_error', '', { duration: 3000 });
          resolve();
        },
      });
    }));
});
