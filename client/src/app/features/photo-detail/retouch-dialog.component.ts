import { Component, ElementRef, computed, effect, inject, input, output, signal, viewChild } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatDialogModule, MatDialogRef, MAT_DIALOG_DATA } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatSliderModule } from '@angular/material/slider';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { firstValueFrom } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';

export interface RetouchDialogData {
  path: string;
  filename: string;
}

interface CropParams {
  x: number;
  y: number;
  width: number;
  height: number;
  unit: 'normalized';
}

interface RetouchParams {
  crop?: CropParams | null;
  rotate: number;
  flip_horizontal: boolean;
  flip_vertical: boolean;
  brightness: number;
  contrast: number;
  saturation: number;
  temperature: number;
  smooth_skin: number;
  whiten_skin: number;
  background_blur: number;
  inpaint_mask_base64?: string | null;
}

interface Spot {
  x: number;
  y: number;
  radius: number;
}

type CropDragHandle = 'move' | 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se';

interface CropDragState {
  handle: CropDragHandle;
  startX: number;
  startY: number;
  startCrop: CropParams;
  imageRect: DOMRect;
}

interface PreviewResponse {
  image_base64: string;
  width: number;
  height: number;
  background_blur_available: boolean;
  mask_provider: string;
}

export interface ApplyResponse {
  output_path: string;
  thumbnail_url: string;
}

const DEFAULT_PARAMS: RetouchParams = {
  crop: null,
  rotate: 0,
  flip_horizontal: false,
  flip_vertical: false,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  temperature: 0,
  smooth_skin: 0,
  whiten_skin: 0,
  background_blur: 0,
  inpaint_mask_base64: null,
};

@Component({
  selector: 'app-retouch-dialog',
  standalone: true,
  imports: [
    FormsModule,
    NgTemplateOutlet,
    MatDialogModule,
    MatButtonModule,
    MatIconModule,
    MatSliderModule,
    MatTabsModule,
    MatTooltipModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    TranslatePipe,
  ],
  template: `
    @if (!embedded()) {
      <h2 mat-dialog-title class="!flex items-center gap-2">
        <mat-icon class="shrink-0">auto_fix_high</mat-icon>
        <span class="truncate">{{ 'retouch.title' | translate }} · {{ activeFilename() }}</span>
      </h2>
    }

    <div [class]="embedded() ? 'retouch-panel' : 'retouch-dialog'">
      <div [class]="embedded() ? 'grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_420px] min-h-0 h-full overflow-hidden' : 'grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px] min-h-[68vh] max-h-[78vh]'">
        <div [class]="embedded() ? 'relative flex items-center justify-center bg-black overflow-hidden min-h-[42vh] lg:min-h-0 h-full' : 'relative flex items-center justify-center bg-black overflow-hidden min-h-[42vh]'">
          @if (loadingPreview()) {
            <div class="absolute inset-0 z-20 grid place-items-center bg-black/35">
              <mat-spinner diameter="36" />
            </div>
          }
          <div [class]="embedded() ? 'relative max-w-full max-h-full h-full flex items-center justify-center' : 'relative max-w-full max-h-full'" (click)="onPreviewClick($event)">
            <img
              #previewImage
              [src]="previewSrc()"
              [alt]="activeFilename()"
              [class]="embedded() ? 'block max-w-full max-h-full object-contain select-none' : 'block max-w-full max-h-[72vh] object-contain select-none'"
              draggable="false"
              [class.cursor-crosshair]="inpaintMode()"
            />
            @if (!cropPreviewConfirmed() && params().crop; as crop) {
              <div
                class="crop-box"
                [style.left.%]="crop.x * 100"
                [style.top.%]="crop.y * 100"
                [style.width.%]="crop.width * 100"
                [style.height.%]="crop.height * 100"
                (pointerdown)="startCropDrag($event, 'move')"
                (click)="$event.stopPropagation()"
                [matTooltip]="'retouch.crop_drag_hint' | translate"
              >
                <span class="crop-rule crop-rule-v crop-rule-v-1"></span>
                <span class="crop-rule crop-rule-v crop-rule-v-2"></span>
                <span class="crop-rule crop-rule-h crop-rule-h-1"></span>
                <span class="crop-rule crop-rule-h crop-rule-h-2"></span>
                <span class="crop-edge crop-edge-n" (pointerdown)="startCropDrag($event, 'n')"></span>
                <span class="crop-edge crop-edge-s" (pointerdown)="startCropDrag($event, 's')"></span>
                <span class="crop-edge crop-edge-e" (pointerdown)="startCropDrag($event, 'e')"></span>
                <span class="crop-edge crop-edge-w" (pointerdown)="startCropDrag($event, 'w')"></span>
                <span class="crop-handle crop-handle-nw" (pointerdown)="startCropDrag($event, 'nw')"></span>
                <span class="crop-handle crop-handle-ne" (pointerdown)="startCropDrag($event, 'ne')"></span>
                <span class="crop-handle crop-handle-sw" (pointerdown)="startCropDrag($event, 'sw')"></span>
                <span class="crop-handle crop-handle-se" (pointerdown)="startCropDrag($event, 'se')"></span>
              </div>
            }
            @for (spot of spots(); track spot.x + ':' + spot.y) {
              <span
                class="absolute rounded-full border-2 border-white/90 bg-red-500/25 pointer-events-none"
                [style.left.%]="spot.x * 100"
                [style.top.%]="spot.y * 100"
                [style.width.px]="spot.radius * 2"
                [style.height.px]="spot.radius * 2"
                [style.marginLeft.px]="-spot.radius"
                [style.marginTop.px]="-spot.radius"
              ></span>
            }
          </div>
        </div>

        <div [class]="embedded() ? 'overflow-y-auto border-l border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface)]' : 'overflow-y-auto border-l border-[var(--mat-sys-outline-variant)] bg-[var(--mat-sys-surface)]'">
          <div class="p-3 border-b border-[var(--mat-sys-outline-variant)] space-y-3">
            @if (embedded()) {
              <div class="grid grid-cols-2 gap-1 rounded-lg bg-[var(--mat-sys-surface-container)] p-1">
                <button
                  type="button"
                  class="h-9 rounded-md inline-flex items-center justify-center gap-2 text-sm transition-colors"
                  (click)="cancelled.emit()"
                >
                  <mat-icon class="!text-base !w-4 !h-4">info</mat-icon>
                  {{ 'photo_detail.details_panel' | translate }}
                </button>
                <button
                  type="button"
                  class="h-9 rounded-md inline-flex items-center justify-center gap-2 text-sm transition-colors bg-[var(--mat-sys-primary-container)] text-[var(--mat-sys-on-primary-container)]"
                >
                  <mat-icon class="!text-base !w-4 !h-4">auto_fix_high</mat-icon>
                  {{ 'retouch.short_title' | translate }}
                </button>
              </div>
            }
            <div class="flex items-center gap-2">
              <button mat-icon-button (click)="undo()" [disabled]="!canUndo()" [matTooltip]="'retouch.undo' | translate">
                <mat-icon>undo</mat-icon>
              </button>
              <button mat-icon-button (click)="redo()" [disabled]="!canRedo()" [matTooltip]="'retouch.redo' | translate">
                <mat-icon>redo</mat-icon>
              </button>
              <button mat-button (click)="reset()">
                <mat-icon>restart_alt</mat-icon>
                {{ 'retouch.reset' | translate }}
              </button>
            </div>
          </div>

          <mat-tab-group mat-stretch-tabs="false" mat-align-tabs="start">
            <mat-tab>
              <ng-template mat-tab-label>
                <mat-icon class="!mr-1">tune</mat-icon>
                {{ 'retouch.basic' | translate }}
              </ng-template>
              <div class="p-4 space-y-5">
                <div class="flex flex-wrap gap-2">
                  <button mat-stroked-button (click)="rotate(-90)"><mat-icon>rotate_left</mat-icon>{{ 'retouch.rotate_left' | translate }}</button>
                  <button mat-stroked-button (click)="rotate(90)"><mat-icon>rotate_right</mat-icon>{{ 'retouch.rotate_right' | translate }}</button>
                  <button mat-stroked-button (click)="toggleFlip('flip_horizontal')"><mat-icon>flip</mat-icon>{{ 'retouch.flip_horizontal' | translate }}</button>
                  <button mat-stroked-button (click)="toggleFlip('flip_vertical')"><mat-icon class="rotate-90">flip</mat-icon>{{ 'retouch.flip_vertical' | translate }}</button>
                </div>
                <label class="flex items-center gap-2 text-sm">
                  <input type="checkbox" [ngModel]="cropEnabled()" (ngModelChange)="setCropEnabled($event)" />
                  {{ 'retouch.enable_crop' | translate }}
                </label>
                @if (cropEnabled()) {
                  <div class="rounded border border-[var(--mat-sys-outline-variant)] p-3 space-y-3">
                    <p class="m-0 text-xs text-[var(--mat-sys-on-surface-variant)]">{{ 'retouch.crop_hint' | translate }}</p>
                    <div class="flex flex-wrap gap-2">
                      <button mat-button (click)="resetCrop()">
                        <mat-icon>crop_free</mat-icon>
                        {{ 'retouch.reset_crop' | translate }}
                      </button>
                      <button mat-stroked-button (click)="confirmCrop()" [disabled]="cropPreviewConfirmed()">
                        <mat-icon>check</mat-icon>
                        {{ 'retouch.confirm_crop' | translate }}
                      </button>
                      <button mat-stroked-button (click)="continueCrop()" [disabled]="!cropPreviewConfirmed()">
                        <mat-icon>crop</mat-icon>
                        {{ 'retouch.continue_crop' | translate }}
                      </button>
                    </div>
                  </div>
                }
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'brightness', label: ('retouch.brightness' | translate), min: -100, max: 100 }" />
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'contrast', label: ('retouch.contrast' | translate), min: -100, max: 100 }" />
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'saturation', label: ('retouch.saturation' | translate), min: -100, max: 100 }" />
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'temperature', label: ('retouch.temperature' | translate), min: -100, max: 100 }" />
              </div>
            </mat-tab>

            <mat-tab>
              <ng-template mat-tab-label>
                <mat-icon class="!mr-1">face_retouching_natural</mat-icon>
                {{ 'retouch.portrait' | translate }}
              </ng-template>
              <div class="p-4 space-y-5">
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'smooth_skin', label: ('retouch.smooth_skin' | translate), min: 0, max: 100 }" />
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'whiten_skin', label: ('retouch.whiten_skin' | translate), min: 0, max: 100 }" />
                <div class="rounded border border-[var(--mat-sys-outline-variant)] p-3 text-xs text-[var(--mat-sys-on-surface-variant)]">
                  {{ 'retouch.skin_mask_note' | translate }}
                </div>
                <div class="space-y-2">
                  <button mat-stroked-button (click)="toggleInpaint()" [class.!bg-red-500]="inpaintMode()" [class.!text-white]="inpaintMode()">
                    <mat-icon>healing</mat-icon>
                    {{ 'retouch.inpaint' | translate }}
                  </button>
                  <button mat-button (click)="clearSpots()" [disabled]="spots().length === 0">{{ 'retouch.clear_spots' | translate }}</button>
                  <p class="text-xs text-[var(--mat-sys-on-surface-variant)]">{{ 'retouch.inpaint_hint' | translate }}</p>
                </div>
              </div>
            </mat-tab>

            <mat-tab>
              <ng-template mat-tab-label>
                <mat-icon class="!mr-1">blur_on</mat-icon>
                {{ 'retouch.background' | translate }}
              </ng-template>
              <div class="p-4 space-y-5">
                <ng-container *ngTemplateOutlet="sliderTpl; context: { key: 'background_blur', label: ('retouch.background_blur' | translate), min: 0, max: 100 }" />
                <div class="rounded border border-[var(--mat-sys-outline-variant)] p-3 text-xs text-[var(--mat-sys-on-surface-variant)]">
                  {{ 'retouch.background_note' | translate }}
                </div>
              </div>
            </mat-tab>

            <mat-tab>
              <ng-template mat-tab-label>
                <mat-icon class="!mr-1">save_as</mat-icon>
                {{ 'retouch.export' | translate }}
              </ng-template>
              <div class="p-4 space-y-4">
                <div class="rounded border border-[var(--mat-sys-primary)] p-3 text-sm">
                  {{ 'retouch.save_copy_note' | translate }}
                </div>
                <button mat-flat-button color="primary" class="w-full" (click)="saveCopy()" [disabled]="saving()">
                  @if (saving()) { <mat-spinner diameter="18" class="!inline-block !mr-2" /> } @else { <mat-icon>save_as</mat-icon> }
                  {{ 'retouch.save_copy' | translate }}
                </button>
              </div>
            </mat-tab>
          </mat-tab-group>
        </div>
      </div>
    </div>

    @if (!embedded()) {
      <mat-dialog-actions align="end">
        <span class="mr-auto text-xs text-[var(--mat-sys-on-surface-variant)]">{{ statusText() }}</span>
        <button mat-button mat-dialog-close>{{ 'ui.buttons.cancel' | translate }}</button>
        <button mat-flat-button color="primary" (click)="saveCopy()" [disabled]="saving()">
          {{ 'retouch.save_copy' | translate }}
        </button>
      </mat-dialog-actions>
    }

    <ng-template #sliderTpl let-key="key" let-label="label" let-min="min" let-max="max">
      <div>
        <div class="flex items-center justify-between text-sm mb-1">
          <span>{{ label }}</span>
          <span class="tabular-nums text-[var(--mat-sys-primary)]">{{ paramValue(key) }}</span>
        </div>
        <mat-slider class="w-full" [min]="min" [max]="max" [step]="1" [discrete]="true">
          <input matSliderThumb [ngModel]="paramValue(key)" (ngModelChange)="setParam(key, $event)" />
        </mat-slider>
      </div>
    </ng-template>
  `,
  styles: [`
    :host {
      display: block;
    }
    .retouch-panel {
      display: block;
      min-height: 0;
      height: 100%;
      overflow: hidden;
    }
    .retouch-dialog {
      display: block;
    }
    .crop-box {
      position: absolute;
      z-index: 10;
      border: 2px solid rgba(255, 255, 255, 0.96);
      box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.46);
      cursor: move;
      touch-action: none;
    }
    .crop-rule {
      position: absolute;
      pointer-events: none;
      background: rgba(255, 255, 255, 0.48);
    }
    .crop-rule-v {
      top: 0;
      bottom: 0;
      width: 1px;
    }
    .crop-rule-h {
      left: 0;
      right: 0;
      height: 1px;
    }
    .crop-rule-v-1 { left: 33.333%; }
    .crop-rule-v-2 { left: 66.666%; }
    .crop-rule-h-1 { top: 33.333%; }
    .crop-rule-h-2 { top: 66.666%; }
    .crop-edge,
    .crop-handle {
      position: absolute;
      z-index: 11;
      touch-action: none;
    }
    .crop-edge-n,
    .crop-edge-s {
      left: 14px;
      right: 14px;
      height: 16px;
      cursor: ns-resize;
    }
    .crop-edge-n { top: -8px; }
    .crop-edge-s { bottom: -8px; }
    .crop-edge-e,
    .crop-edge-w {
      top: 14px;
      bottom: 14px;
      width: 16px;
      cursor: ew-resize;
    }
    .crop-edge-e { right: -8px; }
    .crop-edge-w { left: -8px; }
    .crop-handle {
      width: 18px;
      height: 18px;
      border: 2px solid #fff;
      background: rgba(0, 0, 0, 0.32);
      border-radius: 2px;
    }
    .crop-handle-nw { left: -9px; top: -9px; cursor: nwse-resize; }
    .crop-handle-ne { right: -9px; top: -9px; cursor: nesw-resize; }
    .crop-handle-sw { left: -9px; bottom: -9px; cursor: nesw-resize; }
    .crop-handle-se { right: -9px; bottom: -9px; cursor: nwse-resize; }
  `],
})
export class RetouchDialogComponent {
  private readonly api = inject(ApiService);
  private readonly i18n = inject(I18nService);
  private readonly snackBar = inject(MatSnackBar);
  private readonly dialogRef = inject(MatDialogRef<RetouchDialogComponent>, { optional: true });
  private readonly dialogData = inject<RetouchDialogData | null>(MAT_DIALOG_DATA, { optional: true });

  readonly embedded = input(false);
  readonly imagePath = input<string | null>(null);
  readonly filename = input('');
  readonly saved = output<ApplyResponse>();
  readonly cancelled = output<void>();
  readonly activePath = computed(() => this.imagePath() || this.dialogData?.path || '');
  readonly activeFilename = computed(() => this.filename() || this.dialogData?.filename || '');

  readonly previewImage = viewChild<ElementRef<HTMLImageElement>>('previewImage');
  readonly params = signal<RetouchParams>({ ...DEFAULT_PARAMS });
  readonly previewSrc = signal('');
  readonly loadingPreview = signal(false);
  readonly saving = signal(false);
  readonly inpaintMode = signal(false);
  readonly spots = signal<Spot[]>([]);
  readonly previewWidth = signal(0);
  readonly previewHeight = signal(0);
  readonly statusText = signal('');
  readonly cropPreviewConfirmed = signal(false);

  private undoStack: RetouchParams[] = [];
  private redoStack: RetouchParams[] = [];
  private previewTimer: ReturnType<typeof setTimeout> | null = null;
  private cropDrag: CropDragState | null = null;
  private currentPath = '';

  readonly canUndo = computed(() => this.undoStack.length > 0);
  readonly canRedo = computed(() => this.redoStack.length > 0);
  readonly cropEnabled = computed(() => !!this.params().crop);

  constructor() {
    this.statusText.set(this.i18n.t('retouch.preview_original'));
    effect(() => {
      const path = this.activePath();
      if (!path || path === this.currentPath) return;
      this.currentPath = path;
      this.params.set({ ...DEFAULT_PARAMS });
      this.spots.set([]);
      this.cropPreviewConfirmed.set(false);
      this.undoStack = [];
      this.redoStack = [];
      this.previewWidth.set(0);
      this.previewHeight.set(0);
      this.previewSrc.set(this.api.thumbnailUrl(path, 1920));
      this.statusText.set(this.i18n.t('retouch.preview_original'));
    });
  }

  paramValue(key: keyof RetouchParams): number {
    const value = this.params()[key];
    return typeof value === 'number' ? value : 0;
  }

  setParam(key: keyof RetouchParams, value: number | string): void {
    this.pushHistory();
    this.params.update(p => ({ ...p, [key]: Number(value) }));
    this.schedulePreview();
  }

  rotate(delta: number): void {
    this.pushHistory();
    this.params.update(p => ({ ...p, rotate: (p.rotate + delta + 360) % 360 }));
    this.schedulePreview();
  }

  toggleFlip(key: 'flip_horizontal' | 'flip_vertical'): void {
    this.pushHistory();
    this.params.update(p => ({ ...p, [key]: !p[key] }));
    this.schedulePreview();
  }

  setCropEnabled(enabled: boolean): void {
    this.pushHistory();
    this.cropPreviewConfirmed.set(false);
    this.params.update(p => ({
      ...p,
      crop: enabled ? this.defaultCrop() : null,
    }));
    if (!enabled) this.schedulePreview();
  }

  resetCrop(): void {
    this.pushHistory();
    this.cropPreviewConfirmed.set(false);
    this.params.update(p => ({ ...p, crop: this.defaultCrop() }));
    this.schedulePreview();
  }

  confirmCrop(): void {
    if (!this.params().crop || this.cropPreviewConfirmed()) return;
    this.cropPreviewConfirmed.set(true);
    this.schedulePreview();
  }

  continueCrop(): void {
    if (!this.params().crop || !this.cropPreviewConfirmed()) return;
    this.cropPreviewConfirmed.set(false);
    this.schedulePreview();
  }

  startCropDrag(event: PointerEvent, handle: CropDragHandle): void {
    const crop = this.params().crop;
    const img = this.previewImage()?.nativeElement;
    if (!crop || !img) return;
    event.preventDefault();
    event.stopPropagation();
    this.pushHistory();
    this.cropPreviewConfirmed.set(false);
    this.cropDrag = {
      handle,
      startX: event.clientX,
      startY: event.clientY,
      startCrop: { ...crop },
      imageRect: img.getBoundingClientRect(),
    };
    window.addEventListener('pointermove', this.onCropPointerMove);
    window.addEventListener('pointerup', this.onCropPointerUp, { once: true });
  }

  toggleInpaint(): void {
    this.inpaintMode.update(v => !v);
  }

  onPreviewClick(event: MouseEvent): void {
    if (!this.inpaintMode()) return;
    const img = this.previewImage()?.nativeElement;
    if (!img) return;
    const rect = img.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    this.pushHistory();
    this.spots.update(items => [...items, { x, y, radius: 16 }]);
    this.schedulePreview();
  }

  clearSpots(): void {
    this.pushHistory();
    this.spots.set([]);
    this.schedulePreview();
  }

  undo(): void {
    const prev = this.undoStack.pop();
    if (!prev) return;
    this.redoStack.push(this.cloneParams(this.params()));
    this.params.set(prev);
    this.schedulePreview();
  }

  redo(): void {
    const next = this.redoStack.pop();
    if (!next) return;
    this.undoStack.push(this.cloneParams(this.params()));
    this.params.set(next);
    this.schedulePreview();
  }

  reset(): void {
    this.pushHistory();
    this.params.set({ ...DEFAULT_PARAMS });
    this.spots.set([]);
    this.cropPreviewConfirmed.set(false);
    this.previewSrc.set(this.api.thumbnailUrl(this.activePath(), 1920));
    this.statusText.set(this.i18n.t('retouch.preview_original'));
  }

  async saveCopy(): Promise<void> {
    const path = this.activePath();
    if (!path) return;
    this.saving.set(true);
    try {
      const res = await firstValueFrom(this.api.post<ApplyResponse>('/retouch/apply', {
        image_path: path,
        params: this.paramsWithMask(),
      }));
      this.saved.emit(res);
      this.dialogRef?.close(res);
    } catch {
      this.snackBar.open(this.i18n.t('retouch.save_error'), '', { duration: 3500 });
    } finally {
      this.saving.set(false);
    }
  }

  private pushHistory(): void {
    this.undoStack.push(this.cloneParams(this.params()));
    if (this.undoStack.length > 30) this.undoStack.shift();
    this.redoStack = [];
  }

  private cloneParams(params: RetouchParams): RetouchParams {
    return JSON.parse(JSON.stringify(params));
  }

  private schedulePreview(): void {
    if (this.previewTimer) clearTimeout(this.previewTimer);
    this.previewTimer = setTimeout(() => void this.refreshPreview(), 260);
  }

  private async refreshPreview(): Promise<void> {
    this.loadingPreview.set(true);
    try {
      const res = await firstValueFrom(this.api.post<PreviewResponse>('/retouch/preview', {
        image_path: this.activePath(),
        params: this.paramsWithMask({ includeCrop: this.cropPreviewConfirmed() }),
        max_size: 1280,
      }));
      this.previewSrc.set(res.image_base64);
      this.previewWidth.set(res.width);
      this.previewHeight.set(res.height);
      this.statusText.set(this.i18n.t('retouch.preview_ready'));
    } catch {
      this.statusText.set(this.i18n.t('retouch.preview_error'));
    } finally {
      this.loadingPreview.set(false);
    }
  }

  private paramsWithMask(options: { includeCrop?: boolean } = {}): RetouchParams {
    const mask = this.buildMask();
    const includeCrop = options.includeCrop !== false;
    return { ...this.params(), crop: includeCrop ? this.params().crop : null, inpaint_mask_base64: mask };
  }

  private readonly onCropPointerMove = (event: PointerEvent): void => {
    const drag = this.cropDrag;
    if (!drag) return;
    event.preventDefault();
    const dx = (event.clientX - drag.startX) / Math.max(1, drag.imageRect.width);
    const dy = (event.clientY - drag.startY) / Math.max(1, drag.imageRect.height);
    const crop = this.resizeCrop(drag.startCrop, drag.handle, dx, dy);
    this.params.update(p => ({ ...p, crop }));
  };

  private readonly onCropPointerUp = (): void => {
    this.cropDrag = null;
    window.removeEventListener('pointermove', this.onCropPointerMove);
  };

  private resizeCrop(start: CropParams, handle: CropDragHandle, dx: number, dy: number): CropParams {
    const minSize = 0.05;
    let left = start.x;
    let top = start.y;
    let right = start.x + start.width;
    let bottom = start.y + start.height;

    if (handle === 'move') {
      const width = start.width;
      const height = start.height;
      return {
        x: this.clamp(start.x + dx, 0, 1 - width),
        y: this.clamp(start.y + dy, 0, 1 - height),
        width,
        height,
        unit: 'normalized',
      };
    }

    if (handle.includes('w')) left = this.clamp(start.x + dx, 0, right - minSize);
    if (handle.includes('e')) right = this.clamp(start.x + start.width + dx, left + minSize, 1);
    if (handle.includes('n')) top = this.clamp(start.y + dy, 0, bottom - minSize);
    if (handle.includes('s')) bottom = this.clamp(start.y + start.height + dy, top + minSize, 1);

    return {
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      unit: 'normalized',
    };
  }

  private defaultCrop(): CropParams {
    return { x: 0, y: 0, width: 1, height: 1, unit: 'normalized' };
  }

  private clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
  }

  private buildMask(): string | null {
    const spots = this.spots();
    if (!spots.length) return null;
    const w = this.previewWidth() || 1280;
    const h = this.previewHeight() || 853;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = 'white';
    for (const spot of spots) {
      ctx.beginPath();
      ctx.arc(spot.x * w, spot.y * h, spot.radius * Math.max(w, h) / 900, 0, Math.PI * 2);
      ctx.fill();
    }
    return canvas.toDataURL('image/png');
  }
}
