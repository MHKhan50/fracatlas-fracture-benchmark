// Pure functions used by index.html (tested with Node: node space/test_core.mjs).

// Letterbox geometry. "floor" matches the classifier's Pillow letterbox (longer side = size, centred, black padding);
// "round" matches Ultralytics' LetterBox (pad 114, centred, offsets rounded as round(d - 0.1)).
export function letterboxParams(W, H, size, mode = "floor") {
  const r = Math.min(size / W, size / H);
  const w = Math.max(1, Math.round(W * r)), h = Math.max(1, Math.round(H * r));
  const left = mode === "round" ? Math.max(0, Math.round((size - w) / 2 - 0.1)) : Math.floor((size - w) / 2);
  const top = mode === "round" ? Math.max(0, Math.round((size - h) / 2 - 0.1)) : Math.floor((size - h) / 2);
  return { r, w, h, left, top, W, H };
}

// YOLO11 ONNX output [1, 5, N] (cx, cy, w, h, score), in letterboxed pixels -> boxes in original pixels.
export function decodeYolo(data, N, p, minScore = 0.25) {
  const out = [];
  for (let i = 0; i < N; i++) {
    const s = data[4 * N + i];
    if (s < minScore) continue;
    const cx = data[i], cy = data[N + i], w = data[2 * N + i], h = data[3 * N + i];
    const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
    out.push({
      x1: clip((cx - w / 2 - p.left) / p.r, p.W), y1: clip((cy - h / 2 - p.top) / p.r, p.H),
      x2: clip((cx + w / 2 - p.left) / p.r, p.W), y2: clip((cy + h / 2 - p.top) / p.r, p.H), score: s,
    });
  }
  return out;
}

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy, ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0;
}

// Greedy non-maximum suppression, highest score first.
export function nms(boxes, thr = 0.7, maxBoxes = 10) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score), keep = [];
  for (const b of sorted) {
    if (keep.every((k) => iou(k, b) <= thr)) keep.push(b);
    if (keep.length >= maxBoxes) break;
  }
  return keep;
}

// Calibrated probability, validation threshold and referral rule.
export function verdict(logit, c) {
  const p = 1 / (1 + Math.exp(-logit / c.temperature));
  const conf = Math.max(p, 1 - p);
  if (conf < c.refer_below_confidence)
    return { p, conf, kind: "refer", title: "Unsure: refer",
      text: `Confidence ${conf.toFixed(2)} is below ${c.refer_below_confidence}; the model would pass this image to a clinician.` };
  if (p >= c.threshold)
    return { p, conf, kind: "fracture", title: "Fracture likely", text: `Probability is above the validation threshold (${c.threshold.toFixed(2)}).` };
  return { p, conf, kind: "none", title: "No fracture found", text: `Probability is below the validation threshold (${c.threshold.toFixed(2)}).` };
}
