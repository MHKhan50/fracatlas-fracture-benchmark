import assert from "node:assert/strict";
import { letterboxParams, decodeYolo, nms, iou, verdict } from "./core.mjs";
let n = 0; const t = (name, f) => { f(); n++; console.log("ok", name); };

t("letterbox portrait floor", () => {
  const p = letterboxParams(1000, 2000, 384, "floor");
  assert.equal(p.h, 384); assert.equal(p.w, 192); assert.equal(p.left, 96); assert.equal(p.top, 0);
});
t("letterbox odd padding floor", () => {
  const p = letterboxParams(373, 1000, 384, "floor"); // w = round(143.232) = 143, pad 241 -> left 120
  assert.equal(p.w, 143); assert.equal(p.left, 120);
});
t("letterbox ultralytics round", () => {
  const p = letterboxParams(373, 1000, 640, "round"); // w = round(238.72) = 239, dw = 200.5 -> round(200.4) = 200
  assert.equal(p.w, 239); assert.equal(p.left, 200); assert.equal(p.top, 0);
});
t("decode maps back to original pixels", () => {
  const p = letterboxParams(1000, 2000, 640, "round"); // r = 0.32, w = 320, left = 160
  const N = 3, d = new Float32Array(5 * N);
  // box 0: centre (320, 320), size 64x64, score 0.9 ; box 1 below threshold
  d[0] = 320; d[N] = 320; d[2 * N] = 64; d[3 * N] = 64; d[4 * N] = 0.9; d[4 * N + 1] = 0.1;
  const b = decodeYolo(d, N, p, 0.25);
  assert.equal(b.length, 1);
  assert.ok(Math.abs(b[0].x1 - (288 - 160) / 0.32) < 1e-3); assert.ok(Math.abs(b[0].y2 - 352 / 0.32) < 1e-3);
});
t("decode clips to the image", () => {
  const p = letterboxParams(100, 100, 640, "round");
  const N = 1, d = new Float32Array([5, 5, 40, 40, 0.8]);
  const b = decodeYolo(d, N, p, 0.25)[0];
  assert.equal(b.x1, 0); assert.equal(b.y1, 0);
});
t("iou", () => {
  assert.equal(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 0, y1: 0, x2: 10, y2: 10 }), 1);
  assert.equal(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 20, y1: 20, x2: 30, y2: 30 }), 0);
  assert.ok(Math.abs(iou({ x1: 0, y1: 0, x2: 10, y2: 10 }, { x1: 5, y1: 0, x2: 15, y2: 10 }) - 1 / 3) < 1e-9);
});
t("nms keeps best, drops overlap, keeps distant", () => {
  const k = nms([{ x1: 0, y1: 0, x2: 10, y2: 10, score: 0.5 }, { x1: 1, y1: 0, x2: 11, y2: 10, score: 0.9 },
                 { x1: 50, y1: 50, x2: 60, y2: 60, score: 0.3 }], 0.7);
  assert.equal(k.length, 2); assert.equal(k[0].score, 0.9); assert.equal(k[1].score, 0.3);
});
const c = { temperature: 2, threshold: 0.4, refer_below_confidence: 0.8 };
t("verdict fracture", () => assert.equal(verdict(6, c).kind, "fracture"));
t("verdict none", () => assert.equal(verdict(-6, c).kind, "none"));
t("verdict refer", () => assert.equal(verdict(0.2, c).kind, "refer"));
t("verdict temperature", () => assert.ok(Math.abs(verdict(2, c).p - 1 / (1 + Math.exp(-1))) < 1e-12));
console.log(`${n} tests passed`);
