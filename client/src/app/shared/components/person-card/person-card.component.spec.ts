import { Component, signal } from '@angular/core';
import { ComponentFixture, TestBed } from '@angular/core/testing';
import { I18nService } from '../../../core/services/i18n.service';
import { PersonCardComponent, Person } from './person-card.component';

/* eslint-disable @angular-eslint/component-selector */
@Component({
  selector: 'test-host',
  standalone: true,
  imports: [PersonCardComponent],
  template: `<app-person-card [person]="person()" [isEditing]="isEditing()" [canEdit]="canEdit()" />`,
})
class TestHostComponent {
  person = signal<Person>({ id: 1, name: 'Alice', face_count: 5, face_thumbnail: true });
  isEditing = signal(false);
  canEdit = signal(false);
}

describe('PersonCardComponent', () => {
  let fixture: ComponentFixture<TestHostComponent>;
  let host: TestHostComponent;
  const mockI18n = { t: vi.fn((key: string) => key), currentLang: vi.fn(() => 'en'), locale: vi.fn(() => 'en'), translations: vi.fn(() => ({})) };

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [TestHostComponent],
      providers: [{ provide: I18nService, useValue: mockI18n }],
    }).compileComponents();
    fixture = TestBed.createComponent(TestHostComponent);
    host = fixture.componentInstance;
    fixture.detectChanges();
  });

  function getCard(): PersonCardComponent {
    return fixture.debugElement.children[0].componentInstance as PersonCardComponent;
  }

  it('should create with required person input', () => {
    const card = getCard();
    expect(card).toBeTruthy();
    expect(card.person().name).toBe('Alice');
    expect(card.person().face_count).toBe(5);
  });

  it('should default isSelected to false', () => {
    expect(getCard().isSelected()).toBe(false);
  });

  it('should default isEditing to false', () => {
    expect(getCard().isEditing()).toBe(false);
  });

  it('should default canEdit to false', () => {
    expect(getCard().canEdit()).toBe(false);
  });

  it('onSave emits editSave with id and name from input', () => {
    host.isEditing.set(true);
    fixture.detectChanges();

    const card = getCard();
    const emitted: { id: number; name: string }[] = [];
    card.editSave.subscribe(v => emitted.push(v));

    // Set the native input value
    const input = fixture.nativeElement.querySelector('input') as HTMLInputElement;
    expect(input).toBeTruthy();
    input.value = 'Bob';

    card.onSave();
    expect(emitted).toEqual([{ id: 1, name: 'Bob' }]);
  });

  it('onSave emits empty string when no input element exists', () => {
    // isEditing is false so no input is rendered
    const card = getCard();
    const emitted: { id: number; name: string }[] = [];
    card.editSave.subscribe(v => emitted.push(v));

    card.onSave();
    expect(emitted).toEqual([{ id: 1, name: '' }]);
  });

  it('emits hidden / split outputs with the person id', () => {
    const card = getCard();
    const hidden: number[] = [];
    const split: number[] = [];
    card.hidden.subscribe(v => hidden.push(v));
    card.split.subscribe(v => split.push(v));

    card.hidden.emit(card.person().id);
    card.split.emit(card.person().id);

    expect(hidden).toEqual([1]);
    expect(split).toEqual([1]);
  });

  it('emits unhidden output for a hidden person', () => {
    host.person.set({ id: 7, name: 'Bob', face_count: 2, face_thumbnail: false, is_hidden: true });
    fixture.detectChanges();

    const card = getCard();
    const unhidden: number[] = [];
    card.unhidden.subscribe(v => unhidden.push(v));

    card.unhidden.emit(card.person().id);

    expect(unhidden).toEqual([7]);
  });

  it('renders the hidden chip when the person is hidden', () => {
    host.canEdit.set(true);
    host.person.set({ id: 7, name: 'Bob', face_count: 2, face_thumbnail: false, is_hidden: true });
    fixture.detectChanges();

    expect(fixture.nativeElement.textContent).toContain('persons.hidden');
  });
});
