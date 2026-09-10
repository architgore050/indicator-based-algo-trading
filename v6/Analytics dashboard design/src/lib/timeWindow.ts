/**
 * Centralized datetime tracking for the synced chart system.
 * All panels (price chart, indicators, equity, trade log) share the same
 * [start, end] window. Scrolling/panning the chart or jumping to a trade
 * updates this single source of truth.
 */

export interface TimeWindow {
  start: string; // "YYYY-MM-DD HH:MM:SS"
  end: string;
  startMs: number;
  endMs: number;
}

export const DEFAULT_WINDOW_MINUTES = 180; // 3 hours
const MIN_WINDOW_MINUTES = 30; // prevent zooming to <30m
const MAX_WINDOW_MINUTES = 60 * 24 * 30; // 30 days cap for rendering sanity

function toMs(iso: string): number {
  // "YYYY-MM-DD HH:MM:SS" -> ms
  const t = iso.replace(" ", "T") + "Z";
  const ms = Date.parse(t);
  return isNaN(ms) ? Date.now() : ms;
}

function toIso(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}

export function windowFromBoundsEnd(boundsEnd: string, minutes = DEFAULT_WINDOW_MINUTES): TimeWindow {
  const endMs = toMs(boundsEnd);
  const startMs = endMs - minutes * 60 * 1000;
  return { start: toIso(startMs), end: toIso(endMs), startMs, endMs };
}

export function windowCenteredOn(iso: string, minutes = DEFAULT_WINDOW_MINUTES, bounds?: { start: string; end: string }): TimeWindow {
  const center = toMs(iso);
  const half = (minutes * 60 * 1000) / 2;
  let startMs = center - half;
  let endMs = center + half;
  if (bounds) {
    const b0 = toMs(bounds.start);
    const b1 = toMs(bounds.end);
    const span = endMs - startMs;
    if (startMs < b0) { startMs = b0; endMs = b0 + span; }
    if (endMs > b1) { endMs = b1; startMs = b1 - span; }
  }
  return { start: toIso(startMs), end: toIso(endMs), startMs, endMs };
}

export function panWindow(win: TimeWindow, deltaRatio: number, bounds?: { start: string; end: string }): TimeWindow {
  const span = win.endMs - win.startMs;
  const delta = span * deltaRatio;
  let startMs = win.startMs + delta;
  let endMs = win.endMs + delta;
  if (bounds) {
    const b0 = toMs(bounds.start);
    const b1 = toMs(bounds.end);
    if (startMs < b0) { endMs += b0 - startMs; startMs = b0; }
    if (endMs > b1) { startMs -= endMs - b1; endMs = b1; }
    // clamp again
    if (startMs < b0) startMs = b0;
    if (endMs > b1) endMs = b1;
  }
  return { start: toIso(startMs), end: toIso(endMs), startMs, endMs };
}

export function zoomWindow(win: TimeWindow, factor: number, anchorMs?: number, bounds?: { start: string; end: string }): TimeWindow {
  // factor <1 zoom in (shrink), >1 zoom out (expand)
  const span = win.endMs - win.startMs;
  let newSpan = Math.round(span * factor);
  newSpan = Math.max(MIN_WINDOW_MINUTES * 60 * 1000, Math.min(MAX_WINDOW_MINUTES * 60 * 1000, newSpan));
  const anchor = anchorMs ?? (win.startMs + span / 2);
  const ratio = (anchor - win.startMs) / span;
  let startMs = Math.round(anchor - newSpan * ratio);
  let endMs = startMs + newSpan;
  if (bounds) {
    const b0 = toMs(bounds.start);
    const b1 = toMs(bounds.end);
    if (startMs < b0) { startMs = b0; endMs = b0 + newSpan; }
    if (endMs > b1) { endMs = b1; startMs = b1 - newSpan; }
  }
  return { start: toIso(startMs), end: toIso(endMs), startMs, endMs };
}

export function isTradeInWindow(entryTime: string, win: TimeWindow | null): boolean {
  if (!win) return false;
  const t = entryTime.replace("T", " ").replace(/\.\d+/, "").replace(/Z$/, "").replace(/[+-].*$/, "");
  return t >= win.start && t <= win.end;
}

export function formatWindowLabel(win: TimeWindow | null): string {
  if (!win) return "";
  // "2025-09-11 08:00 — 11:00 UTC"
  const s = win.start.slice(0, 16);
  const eMs = win.endMs;
  const e = new Date(eMs).toISOString().slice(11, 16);
  const date = win.start.slice(0, 10);
  return `${date} ${s.slice(11)} — ${e} UTC · ${Math.round((win.endMs - win.startMs)/60000)}m`;
}
