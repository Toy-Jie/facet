import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { CaptionEditDialogComponent } from './caption-edit-dialog.component';

describe('CaptionEditDialogComponent', () => {
  let component: CaptionEditDialogComponent;
  let mockDialogRef: { close: Mock };
  let mockApi: { put: Mock };

  const dialogData = {
    path: '/photo.jpg',
    filename: 'photo.jpg',
    caption: 'A sunset over the mountains',
  };

  beforeEach(() => {
    mockDialogRef = { close: vi.fn() };
    mockApi = { put: vi.fn(() => of({})) };

    TestBed.configureTestingModule({
      providers: [
        CaptionEditDialogComponent,
        { provide: MatDialogRef, useValue: mockDialogRef },
        { provide: MAT_DIALOG_DATA, useValue: dialogData },
        { provide: ApiService, useValue: mockApi },
      ],
    });
    component = TestBed.inject(CaptionEditDialogComponent);
  });

  it('should initialize with data from dialog', () => {
    expect(component.data.path).toBe('/photo.jpg');
    expect(component.data.filename).toBe('photo.jpg');
    expect(component.captionText).toBe('A sunset over the mountains');
  });

  it('should start with saving false', () => {
    expect(component.saving()).toBe(false);
  });

  it('should call API and close dialog on save', async () => {
    component.captionText = 'Updated caption';

    await component.save();

    expect(mockApi.put).toHaveBeenCalledWith('/caption', {
      path: '/photo.jpg',
      caption: 'Updated caption',
    });
    expect(mockDialogRef.close).toHaveBeenCalledWith('Updated caption');
    expect(component.saving()).toBe(false);
  });

  it('should not close dialog on save error', async () => {
    mockApi.put.mockReturnValue(throwError(() => new Error('Server error')));

    await component.save();

    expect(mockDialogRef.close).not.toHaveBeenCalled();
    expect(component.saving()).toBe(false);
  });

  it('should set saving true during request', async () => {
    let savingDuringRequest = false;
    mockApi.put.mockImplementation(() => {
      savingDuringRequest = component.saving();
      return of({});
    });

    await component.save();

    expect(savingDuringRequest).toBe(true);
  });
});
