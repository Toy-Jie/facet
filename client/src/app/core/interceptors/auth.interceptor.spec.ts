import type { MockedFunction } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { HttpRequest, HttpHandlerFn, HttpResponse } from '@angular/common/http';
import { of } from 'rxjs';
import { authInterceptor } from './auth.interceptor';
import { AuthService } from '../services/auth.service';

describe('authInterceptor', () => {
  let authMock: { token: string | null };
  let next: MockedFunction<HttpHandlerFn>;

  beforeEach(() => {
    authMock = { token: null };
    next = vi.fn().mockReturnValue(of(new HttpResponse({ status: 200 })));

    TestBed.configureTestingModule({
      providers: [{ provide: AuthService, useValue: authMock }],
    });
  });

  const runInterceptor = (req: HttpRequest<unknown>) =>
    TestBed.runInInjectionContext(() => authInterceptor(req, next));

  it('adds Authorization header when token exists', () => {
    authMock.token = 'my-jwt-token';
    const req = new HttpRequest('GET', '/api/photos');

    runInterceptor(req);

    const passedReq = next.mock.calls[0][0];
    expect(passedReq.headers.get('Authorization')).toBe('Bearer my-jwt-token');
  });

  it('does not add header when token is null', () => {
    authMock.token = null;
    const req = new HttpRequest('GET', '/api/photos');

    runInterceptor(req);

    const passedReq = next.mock.calls[0][0];
    expect(passedReq.headers.has('Authorization')).toBe(false);
  });

  it('passes request to next handler', () => {
    authMock.token = null;
    const req = new HttpRequest('GET', '/api/photos');

    runInterceptor(req);

    expect(next).toHaveBeenCalledTimes(1);
  });
});
