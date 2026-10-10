"""
Reference implementation of Manga Flow's classical (OpenCV) panel + bubble detector.

The Android detector (ClassicalPanelDetector.kt / BubbleDetector.kt) is a line-for-line port of
this file. Keeping a Python twin lets the algorithm be evaluated on a desktop against labelled
pages (precision / recall / IoU / reading order) before shipping.

Usage:
    python3 detector_prototype.py            # runs the synthetic benchmark
    python3 detector_prototype.py page.jpg   # prints detections for a real page
"""
import sys
import cv2
import numpy as np

ANALYSIS_LONG_SIDE = 1200


def preprocess(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    scale = ANALYSIS_LONG_SIDE / max(h0, w0)
    scale = min(scale, 1.0)
    img = cv2.resize(img_bgr, (round(w0 * scale), round(h0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    h, w = gray.shape
    # Gutter polarity from the outer frame of the page.
    m = max(4, int(0.02 * min(w, h)))
    frame = np.concatenate([gray[:m].ravel(), gray[-m:].ravel(), gray[:, :m].ravel(), gray[:, -m:].ravel()])
    black_gutters = np.median(frame) < 100
    if black_gutters:
        gray = 255 - gray
    return gray, scale, black_gutters


def gutter_mask(gray):
    """Paper-white area connected to the page edge (after sealing hairline breaks in borders)."""
    h, w = gray.shape
    white = (gray > 215).astype(np.uint8)
    k = max(3, int(0.004 * min(w, h)) | 1)
    white = cv2.erode(white, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ff = white.copy() * 255
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(0, w, 3):
        for y in (0, h - 1):
            if ff[y, x] == 255:
                cv2.floodFill(ff, mask, (x, y), 128)
    for y in range(0, h, 3):
        for x in (0, w - 1):
            if ff[y, x] == 255:
                cv2.floodFill(ff, mask, (x, y), 128)
    g = (ff == 128).astype(np.uint8)
    # Undo the erosion on the gutter so panel boxes are not inflated.
    g = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return g


def xy_cut(g, x0, y0, x1, y1, min_gap, depth=0, out=None):
    """Recursive projection cut on the gutter mask. Tolerates bubbles / art crossing a gutter."""
    if out is None:
        out = []
    sub = g[y0:y1, x0:x1]
    if sub.size == 0:
        return out
    for axis in (1, 0):  # 1: rows (horizontal cuts) first, then columns
        frac = sub.mean(axis=axis)
        is_gap = frac >= 0.80
        runs = []
        i = 0
        n = len(is_gap)
        while i < n:
            if is_gap[i]:
                j = i
                while j < n and is_gap[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        # interior gaps only (not touching region edges) and wide enough
        cuts = [(a, b) for (a, b) in runs if a > 0 and b < n and (b - a) >= min_gap]
        if cuts and depth < 12:
            edges = [0] + [c for ab in cuts for c in ab] + [n]
            for s, e in zip(edges[0::2], edges[1::2]):
                if e - s < min_gap:
                    continue
                if axis == 1:
                    xy_cut(g, x0, y0 + s, x1, y0 + e, min_gap, depth + 1, out)
                else:
                    xy_cut(g, x0 + s, y0, x0 + e, y1, min_gap, depth + 1, out)
            return out
    out.append((x0, y0, x1, y1))
    return out


def detect_panels(gray):
    h, w = gray.shape
    g = gutter_mask(gray)
    min_gap = max(2, int(0.006 * min(w, h)))
    leaves = xy_cut(g, 0, 0, w, h, min_gap)
    page_area = w * h
    cands = []
    for (x0, y0, x1, y1) in leaves:
        content = (1 - g[y0:y1, x0:x1]).astype(np.uint8)
        if content.sum() < 0.006 * page_area:
            continue
        # Split diagonal / irregular arrangements into separate components.
        close_k = max(3, int(0.004 * min(w, h)) | 1)
        content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
        comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.006 * page_area]
        for i in comps:
            comp = (labels == i).astype(np.uint8)
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            eps = 0.01 * cv2.arcLength(hull, True)
            poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2) + [x0, y0]
            bx, by, bw, bh = cv2.boundingRect(c)
            bx += x0
            by += y0
            box = (bx, by, bx + bw, by + bh)
            hull_area = cv2.contourArea(hull)
            solidity = cv2.contourArea(c) / max(hull_area, 1)
            # Gutter evidence: fraction of a ring just outside the box that is gutter.
            r = max(2, min_gap // 2)
            ring = g[max(0, by - r):min(h, by + bh + r), max(0, bx - r):min(w, bx + bw + r)].copy()
            inner = ring[r:-r or None, r:-r or None]
            ring_total = ring.size - inner.size
            ring_g = ring.sum() - inner.sum()
            gutter_ev = float(ring_g) / max(ring_total, 1)
            edge_touch = bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2
            fill = comp.sum() / max(bw * bh, 1)
            score = 0.35 + 0.35 * min(1.0, gutter_ev) + 0.2 * min(1.0, solidity) + 0.1 * min(1.0, fill)
            cands.append(dict(box=box, poly=poly, score=float(min(score, 0.95)), gutter=gutter_ev,
                              borderless=bool(edge_touch and gutter_ev < 0.6)))
    return cands, g


def detect_bubbles(gray, gutter):
    """White, convex, outlined blobs containing small dark glyph-like marks."""
    h, w = gray.shape
    page_area = w * h
    white = ((gray > 200) & (gutter == 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=4)
    bubbles = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.0015 * page_area or area > 0.08 * page_area:
            continue
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, [c], -1, 1, -1)
        filled_area = filled.sum()
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = filled_area / max(hull_area, 1)
        if solidity < 0.80:
            continue
        holes = (filled == 1) & (comp == 0)
        ink = holes.sum() / max(filled_area, 1)
        if ink < 0.01 or ink > 0.40:
            continue
        hn, _, hstats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=8)
        glyphs = [j for j in range(1, hn) if hstats[j, cv2.CC_STAT_AREA] < 0.02 * filled_area]
        if len(glyphs) < 3:
            continue
        # Japanese text: glyph columns are taller than wide in aggregate.
        gx = hstats[glyphs, cv2.CC_STAT_LEFT]
        gy = hstats[glyphs, cv2.CC_STAT_TOP]
        vertical = (gy.max() - gy.min() + 1) >= (gx.max() - gx.min() + 1) * 0.8
        score = min(0.95, 0.4 + 0.3 * solidity + 0.05 * min(len(glyphs), 5))
        bubbles.append(dict(box=(x, y, x + bw, y + bh), score=float(score), vertical=bool(vertical)))
    return bubbles


# ---------------------------------------------------------------- synthetic benchmark

def text_bubble(img, cx, cy, rw, rh, rng):
    cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, 255, -1)
    cv2.ellipse(img, (cx, cy), (rw, rh), 0, 0, 360, 0, 3)
    cols = max(2, rw // 18)
    for k in range(cols):
        x = cx + rw // 2 - k * 18 - 6
        for j in range(max(2, rh // 22)):
            y = cy - rh // 2 + j * 22 + 4
            cv2.rectangle(img, (x, y), (x + 10, y + 12), 0, -1)
    return (cx - rw, cy - rh, cx + rw, cy + rh)


def art(img, x0, y0, x1, y1, rng):
    for _ in range(25):
        p1 = (int(rng.integers(x0, x1)), int(rng.integers(y0, y1)))
        p2 = (int(rng.integers(x0, x1)), int(rng.integers(y0, y1)))
        cv2.line(img, p1, p2, int(rng.integers(0, 120)), int(rng.integers(1, 4)))
    # screentone patch
    tx, ty = int(rng.integers(x0, (x0 + x1) // 2)), int(rng.integers(y0, (y0 + y1) // 2))
    for yy in range(ty, min(y1 - 5, ty + 120), 6):
        for xx in range(tx, min(x1 - 5, tx + 160), 6):
            cv2.circle(img, (xx, yy), 1, 60, -1)


def synth_page(kind, seed):
    if kind == "black_gutter":
        img, truth, bub = synth_page("grid", seed)
        g = img[:, :, 0]
        paper = np.ones_like(g, bool)
        for (x0, y0, x1, y1) in truth:
            paper[y0:y1 + 3, x0:x1 + 3] = False
        img[paper] = 15
        return img, truth, []
    rng = np.random.default_rng(seed)
    W, H = 1600, 2400
    img = np.full((H, W), 255, np.uint8)
    m, gut = 90, 36
    panels, bubbles = [], []
    if kind == "grid":
        rows = [(m, 900), (900 + gut, 1500), (1500 + gut, H - m)]
        for ri, (y0, y1) in enumerate(rows):
            ncol = [2, 3, 1][ri]
            cw = (W - 2 * m - gut * (ncol - 1)) // ncol
            for ci in range(ncol):
                x0 = W - m - (ci + 1) * cw - ci * gut  # right to left
                panels.append((x0, y0, x0 + cw, y1))
    elif kind == "tall_right":
        panels = [(W // 2 + gut // 2, m, W - m, H - m), (m, m, W // 2 - gut // 2, H // 2 - gut // 2),
                  (m, H // 2 + gut // 2, W // 2 - gut // 2, H - m)]
    elif kind == "full":
        panels = [(m, m, W - m, H - m)]
    elif kind == "borderless":
        panels = [(0, 0, W, 1100), (m, 1100 + gut, W // 2 - gut // 2, H - m), (W // 2 + gut // 2, 1100 + gut, W - m, H - m)]
    elif kind == "diagonal":
        panels = [(m, m, W - m, 800)]
        # two panels separated by a diagonal gutter
        a = np.array([[W - m, 800 + gut], [W - m, H - m], [900, H - m], [700, 800 + gut]])
        b = np.array([[m, 800 + gut], [700 - gut, 800 + gut], [900 - gut, H - m], [m, H - m]])
        for poly in (a, b):
            layer = np.full_like(img, 235)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            art(layer, x0 + 40, y0 + 40, x1 - 40, y1 - 40, rng)
            pm = np.zeros_like(img)
            cv2.fillPoly(pm, [poly.astype(np.int32)], 1)
            img[pm == 1] = layer[pm == 1]  # artwork clipped to the panel shape
            cv2.polylines(img, [poly.astype(np.int32)], True, 0, 5)
        truth_extra = [tuple(a.min(0)) + tuple(a.max(0)), tuple(b.min(0)) + tuple(b.max(0))]
    for (x0, y0, x1, y1) in panels:
        cv2.rectangle(img, (x0, y0), (x1, y1), 235, -1)
        if not (kind == "borderless" and y0 == 0):
            cv2.rectangle(img, (x0, y0), (x1, y1), 0, 5)
        else:
            img[y0:y1, x0:x1] = 200  # full-bleed art reaching the page edge
        art(img, x0 + 20, y0 + 20, x1 - 20, y1 - 20, rng)
        if (x1 - x0) > 300 and (y1 - y0) > 300:
            bw, bh = 70, 110
            bubbles.append(text_bubble(img, x1 - 140, y0 + 160, bw, bh, rng))
    truth = list(panels)
    if kind == "diagonal":
        truth += truth_extra
    if kind == "grid":
        # A bubble breaking out of the first panel across the gutter
        bubbles.append(text_bubble(img, panels[0][0] + 200, panels[0][3], 80, 100, rng))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), truth, bubbles


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def evaluate():
    tp = fp = fn = 0
    ious = []
    btp = bfn = 0
    for kind in ["grid", "tall_right", "full", "borderless", "diagonal", "black_gutter"]:
        for seed in range(4):
            img, truth, tbubbles = synth_page(kind, seed)
            gray, s, black = preprocess(img)
            cands, g = detect_panels(gray)
            boxes = [tuple(v / s for v in c["box"]) for c in cands]
            matched = set()
            for t in truth:
                best = max(range(len(boxes)), key=lambda i: iou(t, boxes[i]), default=None)
                if best is not None and iou(t, boxes[best]) >= 0.75 and best not in matched:
                    matched.add(best)
                    tp += 1
                    ious.append(iou(t, boxes[best]))
                else:
                    fn += 1
                    print(f"  miss {kind}/{seed} truth={t} got={[tuple(int(v) for v in b) for b in boxes]}")
            fp += len(boxes) - len(matched)
            bb = [tuple(v / s for v in b["box"]) for b in detect_bubbles(gray, g)]
            for t in tbubbles:
                if any(iou(t, b) >= 0.5 for b in bb):
                    btp += 1
                else:
                    bfn += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    print(f"panels: precision={p:.3f} recall={r:.3f} meanIoU={np.mean(ious):.3f}  (tp={tp} fp={fp} fn={fn})")
    print(f"bubbles: recall={btp / max(btp + bfn, 1):.3f} ({btp}/{btp + bfn})")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        gray, s, black = preprocess(img)
        cands, g = detect_panels(gray)
        for c in cands:
            print([int(v / s) for v in c["box"]], round(c["score"], 2), "borderless" if c["borderless"] else "")
        for b in detect_bubbles(gray, g):
            print("bubble", [int(v / s) for v in b["box"]], round(b["score"], 2))
    else:
        evaluate()
