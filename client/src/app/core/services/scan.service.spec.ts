import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { ScanService, ScanStatus } from './scan.service';
import { AuthService } from './auth.service';

class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  close(): void {
    this.closed = true;
  }

  simulateMessage(data: ScanStatus): void {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent);
  }

  simulateError(): void {
    this.onerror?.();
  }
}

describe('ScanService', () => {
  let service: ScanService;
  let httpTesting: HttpTestingController;
  let originalEventSource: typeof EventSource;

  const mockStatus: ScanStatus = {
    running: true,
    directories: ['/photos'],
    output: ['Processing...'],
    elapsed_seconds: 5,
    exit_code: null,
  };

  beforeEach(() => {
    MockEventSource.instances = [];
    originalEventSource = globalThis.EventSource;
    globalThis.EventSource = MockEventSource as unknown as typeof EventSource;

    vi.spyOn(Storage.prototype, 'getItem').mockReturnValue('test-token');

    TestBed.configureTestingModule({
      providers: [
        ScanService,
        AuthService,
        provideHttpClient(),
        provideHttpClientTesting(),
      ],
    });
    service = TestBed.inject(ScanService);
    httpTesting = TestBed.inject(HttpTestingController);
  });

  afterEach(() => {
    service.ngOnDestroy();
    httpTesting.verify();
    globalThis.EventSource = originalEventSource;
    vi.restoreAllMocks();
  });

  describe('initial state', () => {
    it('should have idle status', () => {
      expect(service.status().running).toBe(false);
      expect(service.status().output).toEqual([]);
    });

    it('should not be connected', () => {
      expect(service.connected()).toBe(false);
    });
  });

  describe('connect()', () => {
    it('should create an EventSource with token query param', () => {
      service.connect();

      expect(MockEventSource.instances).toHaveLength(1);
      expect(MockEventSource.instances[0].url).toContain('/api/scan/stream?');
      expect(MockEventSource.instances[0].url).toContain('token=test-token');
      expect(MockEventSource.instances[0].url).toContain('lines=50');
      expect(service.connected()).toBe(true);
    });

    it('should not connect when no token is available', () => {
      vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(null);
      service.connect();

      expect(MockEventSource.instances).toHaveLength(0);
      expect(service.connected()).toBe(false);
    });

    it('should update status when receiving a message', () => {
      service.connect();
      MockEventSource.instances[0].simulateMessage(mockStatus);

      expect(service.status()).toEqual(mockStatus);
    });

    it('should disconnect when scan finishes', () => {
      service.connect();
      const source = MockEventSource.instances[0];

      source.simulateMessage({ ...mockStatus, running: false, exit_code: 0 });

      expect(source.closed).toBe(true);
      expect(service.connected()).toBe(false);
    });

    it('should close previous connection when connecting again', () => {
      service.connect();
      const first = MockEventSource.instances[0];

      service.connect();
      expect(first.closed).toBe(true);
      expect(MockEventSource.instances).toHaveLength(2);
    });
  });

  describe('connect() SSE error fallback', () => {
    it('should fall back to polling on SSE error', async () => {
      service.connect();
      const source = MockEventSource.instances[0];

      source.simulateError();

      expect(source.closed).toBe(true);
      expect(service.connected()).toBe(false);

      await Promise.resolve();

      const req = httpTesting.expectOne((r) => r.url === '/api/scan/status');
      expect(req.request.params.get('lines')).toBe('50');
      req.flush(mockStatus);

      await Promise.resolve();

      expect(service.status()).toEqual(mockStatus);
    });
  });

  describe('disconnect()', () => {
    it('should close the EventSource', () => {
      service.connect();
      const source = MockEventSource.instances[0];

      service.disconnect();

      expect(source.closed).toBe(true);
      expect(service.connected()).toBe(false);
    });
  });

  describe('startScan()', () => {
    it('should POST to /scan/start and then connect SSE', async () => {
      const promise = service.startScan(['/photos']);

      const req = httpTesting.expectOne('/api/scan/start');
      expect(req.request.method).toBe('POST');
      expect(req.request.body).toEqual({ directories: ['/photos'] });
      req.flush({ success: true });

      await promise;

      expect(MockEventSource.instances).toHaveLength(1);
      expect(service.connected()).toBe(true);
    });
  });

  describe('loadDirectories()', () => {
    it('should fetch configured directories', async () => {
      const promise = service.loadDirectories();

      const req = httpTesting.expectOne('/api/scan/directories');
      expect(req.request.method).toBe('GET');
      req.flush({ directories: [{ path: '/photos', owner: 'shared' }] });

      const dirs = await promise;
      expect(dirs).toEqual([{ path: '/photos', owner: 'shared' }]);
    });
  });
});
