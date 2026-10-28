"""Scene environment for synthetic dataset v5 (generate_v5.py): a camera per image, a procedural floor per image and
warehouse surroundings that are NOT shipping units (never labelled).

Nothing here is taken from cam_000CDF0AD90F (its frames are the test set): no real floor image, not its camera
model (51185017), no tone / noise / sun-patch matching to its frames.

Camera   one of the three JAI GO(DA)-5000M + KOWA lenses of the shipped Dachser models (LM8HC 51185016, LM12HC
         51185020, LM16HC 51185012; median factory calibration of the model + unit-to-unit jitter), mounted at a
         height typical of that lens
         (LM8 2.6-4.0 m, LM12 3.9-6.3 m, LM16 5.3-8.0 m), looking straight down; images framed like prepare_real.py
         (undistorted, x0.5 -> 1280x1024, black corners where the undistorted frame has no raw pixels)
Floor    image-space albedo map (2x supersampled) of a floor that is rendered (lit, shadowed, glossy) by Cycles:
         concrete (light / worn / polished), painted epoxy, asphalt, dark painted, steel / tiles; slab joints,
         cracks, repairs, stains, wet patches, tyre marks, wear paths; markings (lanes, pallet bays, hatched zones,
         arrows, zone codes, tape) with wear; dock levellers; the bay / yard / trailer beyond a dock wall
Things   walls with dock doors (doors open / half / closed, door seals, bollards), columns (concrete / steel, with
         guards), guard rails and barriers, pallet racks, roller / belt conveyors, forklifts, electric pallet trucks,
         people (hi-vis, helmets, walking, carrying), empty pallets and pallet stacks, bins, cones, bollards, cables,
         cardboard sheets, film rolls, platform trolleys, desks, overhead pipes

All builders return Thing(parts, quads, yaw, cx, cy, foot) in the v2 world frame (x right, y down = image axes,
z = height above the floor; camera at (0, 0, mount_h)). `foot` = convex footprint (k, 2) used to keep units free.
"""
import csv
import math
import os
import statistics
from dataclasses import dataclass, field

import cv2
import numpy as np

import generate_v2 as G
import units_v3 as U
from camera import Camera

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CALIB = os.path.join(os.path.dirname(ROOT), "cam_models_calib", "2025.10.23_Dachser_Shipped.txt")
ALB = 220.0  # same as generate_v3.ALB: tone of a white (albedo 1) surface under the nominal light
FULL = (2560, 2048)  # JAI GO-5000M sensor
# lens -> (shipped model whose calibrations are used, mounting heights, share of images)
LENSES = {"LM8HC": ("51185016", (2.6, 4.0), 0.2), "LM12HC": ("51185020", (3.9, 6.3), 0.4),
          "LM16HC": ("51185012", (5.3, 8.0), 0.4)}
_CAL = {}


# ---------------------------------------------------------------------------------- camera
def lens_calibration(model):
    """Median factory intrinsics of one shipped camera model (full resolution)."""
    if model not in _CAL:
        rows = [r for r in csv.DictReader(open(CALIB), delimiter="\t") if r["Model"] == model and r.get("fx")]
        _CAL[model] = {k: statistics.median(float(r[k]) for r in rows)
                       for k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")}
    return dict(_CAL[model])


def make_camera(rng, scale=0.5, lens=None, mount_h=None):
    """Random camera: lens, unit-to-unit calibration jitter, mounting height -> Camera (+ .lens, .valid)."""
    if lens is None:
        names = list(LENSES)
        lens = names[rng.choice(len(names), p=[LENSES[n][2] for n in names])]
    model, (h0, h1), _ = LENSES[lens]
    ci = lens_calibration(model)
    f = rng.normal(1.0, 0.004)
    ci["fx"], ci["fy"] = ci["fx"] * f, ci["fy"] * f * rng.normal(1.0, 0.0005)
    ci["cx"], ci["cy"] = ci["cx"] + rng.normal(0, 8), ci["cy"] + rng.normal(0, 8)
    for k in ("k1", "k2", "k3"):
        ci[k] *= rng.normal(1.0, 0.02)
    return camera_from(ci, mount_h if mount_h is not None else rng.uniform(h0, h1), lens, scale)


def camera_from(ci, mount_h, lens, scale=0.5):
    cam = Camera.__new__(Camera)
    cam.full_size, cam.scale = FULL, scale
    cam.size = (round(FULL[0] * scale), round(FULL[1] * scale))
    cam.fx, cam.fy = ci["fx"] * scale, ci["fy"] * scale
    cam.cx, cam.cy = (ci["cx"] + 0.5) * scale - 0.5, (ci["cy"] + 0.5) * scale - 0.5
    cam.K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    cam.dist = np.array([ci["k1"], ci["k2"], ci["p1"], ci["p2"], ci["k3"]])
    cam.mount_h, cam.lens, cam.full_intrinsics = float(mount_h), lens, ci
    return cam


def valid_mask(cam):
    """Pixels of the undistorted, downscaled frame that have raw pixels behind them (prepare_real.py framing)."""
    ci = cam.full_intrinsics
    K = np.array([[ci["fx"], 0, ci["cx"]], [0, ci["fy"], ci["cy"]], [0, 0, 1]])
    ones = cv2.undistort(np.full((FULL[1], FULL[0]), 255, np.uint8), K, cam.dist)
    return cv2.resize(ones, cam.size, interpolation=cv2.INTER_AREA) > 127


def view_extent(cam, z=0.0):
    """World (x0, y0, x1, y1) seen on the plane at height z."""
    d = cam.mount_h - z
    return (-cam.cx * d / cam.fx, -cam.cy * d / cam.fy, (cam.size[0] - 1 - cam.cx) * d / cam.fx,
            (cam.size[1] - 1 - cam.cy) * d / cam.fy)


def project_world(cam, P):
    P = np.asarray(P, np.float64).reshape(-1, 3)
    d = cam.mount_h - P[:, 2]
    return np.stack([cam.fx * P[:, 0] / d + cam.cx, cam.fy * P[:, 1] / d + cam.cy], 1)


# ---------------------------------------------------------------------------------- geometry helpers
@dataclass
class Thing:
    parts: list
    quads: list = field(default_factory=list)
    yaw: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    foot: np.ndarray = None  # (k, 2) world footprint
    films: list = field(default_factory=list)


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def world_rect(cx, cy, l, w, yaw):
    """Rectangle footprint in the world frame, same convention as G.transform (x along l)."""
    loc = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]])
    return G.transform(np.c_[loc, np.zeros(4)], yaw, cx, cy)[:, :2]


def poly_gap(A, B):
    """Separation of two convex polygons along the best separating axis (< 0: they overlap)."""
    best = -np.inf
    for P in (A, B):
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]
            n = np.array([e[1], -e[0]])
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n /= nn
            a, b = A @ n, B @ n
            best = max(best, b.min() - a.max(), a.min() - b.max())
    return best


def free(foot, taken, gap=0.0):
    return all(poly_gap(foot, t) > gap for t in taken)


ENV_TEX = 384  # px, longest side of an environment texture (plain surfaces; keeps the atlas small)


def capped(fn, cap=ENV_TEX):
    """Texture function that never makes tiles larger than cap (the tile is stretched over the face)."""
    def f(fi, w, h):
        k = min(1.0, cap / max(w, h))
        return fn(fi, max(4, int(w * k)), max(4, int(h * k)))
    return f


def cap_parts(parts):
    for p in parts:
        if "tex" in p and not getattr(p["tex"], "_capped", False):
            p["tex"] = capped(p["tex"])
            p["tex"]._capped = True
        if "surf" in p and p.get("key") is None:  # smooth env surfaces: share one small tile per part
            p["key"] = f"env_{id(p)}"
    return parts


def tex_plain(rng, tone, noise=3.0, mottle=5.0):
    return lambda f, w, h: G.plain(rng, w, h, float(tone), noise, mottle)


def tex_hazard(rng, a=None, b=None, period=0.12):
    """Yellow / black diagonal stripes (mono)."""
    a = rng.uniform(130, 175) if a is None else a
    b = rng.uniform(15, 35) if b is None else b

    def f(fi, w, h):
        yy, xx = np.mgrid[0:h, 0:w]
        s = ((xx + yy) // max(2, G.px(period))) % 2
        return np.clip(np.where(s == 0, a, b).astype(np.float32) + G.fine(rng, h, w, 3), 0, 255)
    return f


def tex_bands(rng, a, b, n_bands=3):
    """Bands along the texture rows (bollards / posts with dark rings)."""
    def f(fi, w, h):
        t = np.full((h, w), a, np.float32)
        for k in range(n_bands):
            y0 = int(h * (0.15 + 0.7 * k / max(1, n_bands - 1)))
            t[max(0, y0 - h // 20):y0 + h // 20] = b
        return np.clip(t + G.fine(rng, h, w, 3), 0, 255)
    return f


def box_p(cx, cy, l, w, z0, z1, tex, mat="paint", yaw=0.0):
    return G.part(G.prism(cx, cy, l, w, z0, z1, yaw), tex, mat=mat)


def beam(a, b, t_w, t_h, tex, mat="steel"):
    """Straight bar between 3D points a, b with a (t_w x t_h) cross-section; 'up' of the section ~ +z."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = (b - a) / 2
    ln = np.linalg.norm(d)
    u = d / max(ln, 1e-9)
    ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(ref, u)
    side /= np.linalg.norm(side)
    up = np.cross(u, side)
    return G.part(G.oriented_box((a + b) / 2, d, side * t_w / 2, up * t_h / 2), tex, mat=mat)


def cylinder(x, y, r, z0, z1, tex, mat="paint", n=16):
    prof = [(0.0, 0.0), (r, 0.0), (r, z1 - z0), (0.0, z1 - z0)]
    return U.lathe_at(prof, tex, mat, t=(x, y, z0), n=n)


def sphere(x, y, z, r, tex, mat="matte", n=16, flat=0.0):
    ph = np.linspace(-math.pi / 2, math.pi / 2, 9)
    prof = [(r * math.cos(p), r * math.sin(p) * (1 - flat)) for p in ph]
    prof[0], prof[-1] = (0.0, prof[0][1]), (0.0, prof[-1][1])
    return U.lathe_at(prof, tex, mat, t=(x, y, z), n=n, sharp=0)


# ---------------------------------------------------------------------------------- floor
FLOOR_KINDS = {  # name: (share, base tone range, roughness range)
    "concrete": (0.30, (80, 160), (0.8, 0.95)),
    "worn_concrete": (0.17, (60, 120), (0.85, 0.95)),
    "polished": (0.10, (100, 170), (0.3, 0.55)),
    "epoxy": (0.15, (40, 140), (0.2, 0.45)),
    "dark_painted": (0.13, (25, 60), (0.5, 0.8)),
    "asphalt": (0.07, (25, 65), (0.9, 0.98)),
    "tiles": (0.08, (90, 190), (0.4, 0.8)),
}


class FloorPainter:
    """Image-space floor texture (2x the render resolution); world <-> texture pixel mapping of the floor plane."""

    def __init__(self, rng, cam, ss=2):
        self.rng, self.cam, self.ss = rng, cam, ss
        self.h, self.w = cam.size[1] * ss, cam.size[0] * ss
        self.mpp = cam.mount_h / cam.fx / ss  # metres per texture pixel

    def to_px(self, xy):
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        c = self.cam
        return np.stack([(c.fx * xy[:, 0] / c.mount_h + c.cx + 0.5) * self.ss - 0.5,
                         (c.fy * xy[:, 1] / c.mount_h + c.cy + 0.5) * self.ss - 0.5], 1)

    def L(self, m):
        return max(1, int(round(m / self.mpp)))

    def noise(self, cell_m):
        return G.smooth_noise(self.rng, self.h, self.w, max(2, self.L(cell_m)))

    def mask(self):
        return np.zeros((self.h, self.w), np.float32)

    def line(self, m, a, b, width_m, val=1.0):
        p = np.round(self.to_px([a, b]) * 4).astype(np.int32)
        cv2.line(m, tuple(p[0]), tuple(p[1]), float(val), max(1, self.L(width_m)), cv2.LINE_AA, shift=2)

    def poly(self, m, P, val=1.0):
        cv2.fillPoly(m, [np.round(self.to_px(P) * 4).astype(np.int32)], float(val), cv2.LINE_AA, shift=2)

    def polyline(self, m, P, width_m, val=1.0, closed=False):
        cv2.polylines(m, [np.round(self.to_px(P) * 4).astype(np.int32)], closed, float(val),
                      max(1, self.L(width_m)), cv2.LINE_AA, shift=2)

    def text(self, m, s, center, height_m, ang):
        """Painted characters (zone codes / numbers) lying on the floor."""
        font = cv2.FONT_HERSHEY_SIMPLEX if self.rng.random() < 0.6 else cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(s, font, 4.0, 10)
        img = np.zeros((th + 40, tw + 40), np.uint8)
        cv2.putText(img, s, (20, th + 20), font, 4.0, 255, 10, cv2.LINE_AA)
        img = img.astype(np.float32) / 255
        k = height_m / th  # metres per glyph pixel
        hw, hh = img.shape[1] / 2 * k, img.shape[0] / 2 * k
        loc = np.array([[-hw, -hh], [hw, -hh], [hw, hh]])
        dst = self.to_px(np.asarray(center) + loc @ rot2(ang).T)
        A = cv2.getAffineTransform(np.float32([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]]]),
                                   np.float32(dst))
        np.maximum(m, cv2.warpAffine(img, A, (self.w, self.h)), out=m)


def worn(rng, P, m, amount):
    """Paint wear: break up a marking mask with blotchy noise and scratches."""
    if amount <= 0:
        return m
    n = P.noise(rng.uniform(0.05, 0.3)) * 0.5 + P.noise(0.02) * 0.5
    return m * np.clip(1.0 - amount * 2.5 * np.clip(n + 0.3 * amount, 0, None), 0, 1)


def make_floor(rng, cam, layout):
    """-> (albedo (h, w) float32 in [0, 1.1], roughness, info). layout: axis angle, rows / bays from the scene."""
    P = FloorPainter(rng, cam)
    names = list(FLOOR_KINDS)
    kind = names[rng.choice(len(names), p=np.array([FLOOR_KINDS[k][0] for k in names]) / sum(
        FLOOR_KINDS[k][0] for k in names))]
    _, (b0, b1), (r0, r1) = FLOOR_KINDS[kind]
    base, rough = rng.uniform(b0, b1), rng.uniform(r0, r1)
    h, w = P.h, P.w
    t = np.full((h, w), base, np.float32)
    t += P.noise(rng.uniform(0.6, 3.0)) * rng.uniform(2, 12)               # large mottling
    t += P.noise(rng.uniform(0.08, 0.3)) * rng.uniform(1, 6)               # trowel / roller marks
    t += rng.normal(0, rng.uniform(1.0, 4.0), (h, w)).astype(np.float32)  # grain
    ax = layout["axis"]
    R = rot2(ax)
    if kind == "asphalt":  # aggregate speckle
        sp = (rng.random((h, w)) < 0.04).astype(np.float32)
        t += cv2.GaussianBlur(sp, (0, 0), max(0.6, P.L(0.004))) * rng.uniform(40, 90) * rng.choice([-1, 1])
    if kind == "tiles":  # tiles / steel plates grid
        s = rng.uniform(0.3, 1.2)
        m = P.mask()
        for k in range(-40, 41):
            for d in (0, 1):
                a = R @ (np.array([k * s, -25.0]) if d == 0 else np.array([-25.0, k * s]))
                b = R @ (np.array([k * s, 25.0]) if d == 0 else np.array([25.0, k * s]))
                P.line(m, a, b, rng.uniform(0.003, 0.008))
        t += m * rng.uniform(-35, -10)
        t += np.sign(P.noise(s)) * rng.uniform(0, 6)
    if kind in ("concrete", "worn_concrete", "polished", "epoxy") and rng.random() < 0.8:  # slab / saw-cut joints
        s1, s2 = rng.uniform(3.0, 7.0), rng.uniform(3.0, 7.0)
        o1, o2 = rng.uniform(0, s1), rng.uniform(0, s2)
        m = P.mask()
        wj = rng.uniform(0.004, 0.012)
        for k in range(-6, 7):
            P.line(m, R @ np.array([o1 + k * s1, -30.0]), R @ np.array([o1 + k * s1, 30.0]), wj)
            P.line(m, R @ np.array([-30.0, o2 + k * s2]), R @ np.array([30.0, o2 + k * s2]), wj)
        t += m * rng.uniform(-45, -15)
    if kind != "tiles":
        for _ in range(rng.poisson(1.5 if "concrete" in kind else 0.5)):  # cracks
            p0 = rng.uniform(-4, 4, 2)
            pts = [p0]
            a = rng.uniform(0, 2 * math.pi)
            for _ in range(int(rng.integers(6, 30))):
                a += rng.normal(0, 0.5)
                pts.append(pts[-1] + rng.uniform(0.03, 0.15) * np.array([math.cos(a), math.sin(a)]))
            m = P.mask()
            P.polyline(m, np.array(pts), rng.uniform(0.002, 0.005))
            t += m * rng.uniform(-50, -15)
    for _ in range(rng.poisson(1.0)):  # repaired / replaced patches
        c, l, wd = rng.uniform(-4, 4, 2), rng.uniform(0.3, 2.5), rng.uniform(0.3, 2.0)
        m = P.mask()
        P.poly(m, world_rect(c[0], c[1], l, wd, ax + rng.normal(0, 0.05)))
        t += cv2.GaussianBlur(m, (0, 0), 1.0) * rng.uniform(-25, 25)
    for _ in range(rng.poisson(2.5 if kind != "polished" else 1.0)):  # stains, oil, wet patches
        c = rng.uniform(-5, 5, 2)
        m = P.mask()
        r = rng.uniform(0.05, 0.9)
        cv2.ellipse(m, tuple(int(v) for v in P.to_px(c)[0]), (P.L(r), P.L(r * rng.uniform(0.3, 1.0))),
                    rng.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), P.L(r * rng.uniform(0.1, 0.5)) + 1) * np.clip(1 + P.noise(r / 2), 0, 1.5)
        t += m * (rng.uniform(-40, -5) if rng.random() < 0.8 else rng.uniform(5, 20))
    for _ in range(rng.poisson(1.5)):  # tyre marks of forklifts / trucks: pairs of arcs
        c = rng.uniform(-6, 6, 2)
        rad = rng.uniform(2, 15)
        a0 = rng.uniform(0, 2 * math.pi)
        arc = np.linspace(a0, a0 + rng.uniform(0.3, 1.2) * rng.choice([-1, 1]), 40)
        track = rng.uniform(0.8, 1.1)
        m = P.mask()
        for dr in (0, track):
            pts = c + (rad + dr) * np.stack([np.cos(arc), np.sin(arc)], 1)
            P.polyline(m, pts, rng.uniform(0.06, 0.2))
        t += cv2.GaussianBlur(m, (0, 0), 2) * rng.uniform(-22, -4) * np.clip(1 + P.noise(0.4), 0, 2)
    if rng.random() < 0.4:  # wear path along the main lane (lighter or darker)
        m = P.mask()
        off = rng.uniform(-2, 2)
        P.poly(m, (R @ np.array([[-30, off - 0.8], [30, off - 0.8], [30, off + 0.8], [-30, off + 0.8]]).T).T)
        t += cv2.GaussianBlur(m, (0, 0), P.L(0.4)) * rng.uniform(-15, 15)

    # ---- markings
    paint = P.mask()

    def paint_tone():
        r = rng.random()
        return rng.uniform(175, 215) if r < 0.45 else rng.uniform(120, 165) if r < 0.85 else \
            rng.uniform(15, 40) if base > 80 else rng.uniform(150, 200)

    marks = []  # (mask, tone)
    if rng.random() < 0.75:
        lw = rng.uniform(0.05, 0.15)
        tone = paint_tone()
        m = P.mask()
        style = rng.choice(["lanes", "bays", "both"], p=[0.35, 0.35, 0.3])
        if style in ("lanes", "both"):  # lane lines (solid / dashed)
            sp = rng.uniform(2.0, 5.0)
            off = rng.uniform(0, sp)
            dashed = rng.random() < 0.3
            for k in range(-4, 5):
                y = off + k * sp
                if dashed:
                    dl, gap = rng.uniform(0.5, 1.5), rng.uniform(0.3, 1.0)
                    for x in np.arange(-15, 15, dl + gap):
                        P.line(m, R @ [x, y], R @ [x + dl, y], lw)
                else:
                    P.line(m, R @ [-30, y], R @ [30, y], lw)
        if style in ("bays", "both"):  # pallet bays: comb / rectangles (like the dach018 floor)
            bw, bd = rng.uniform(0.9, 1.5), rng.uniform(1.2, 2.8)
            y0 = rng.uniform(-3, 3)
            x0 = rng.uniform(0, bw)
            closed = rng.random() < 0.5
            for k in range(-12, 13):
                x = x0 + k * bw
                P.line(m, R @ [x, y0], R @ [x, y0 + bd], lw)
            P.line(m, R @ [-30, y0], R @ [30, y0], lw)
            if closed:
                P.line(m, R @ [-30, y0 + bd], R @ [30, y0 + bd], lw)
            layout["bays"] = (y0, bd, bw, x0)
            if rng.random() < 0.5:  # numbers in the bays
                for k in range(-12, 13):
                    c = R @ [x0 + (k + 0.5) * bw, y0 + bd * rng.uniform(0.15, 0.3)]
                    P.text(m, str(int(rng.integers(1, 99))), c, rng.uniform(0.15, 0.3), ax)
        marks.append((worn(rng, P, m, rng.uniform(0, 0.8) if rng.random() < 0.7 else 0.0), tone))
    if rng.random() < 0.25:  # hatched no-parking zone
        m = P.mask()
        c, l, wd = rng.uniform(-3, 3, 2), rng.uniform(1.0, 4.0), rng.uniform(0.8, 3.0)
        zone = world_rect(c[0], c[1], l, wd, ax)
        P.polyline(m, zone, 0.08, closed=True)
        inner = P.mask()
        P.poly(inner, zone)
        st = P.mask()
        for k in range(-60, 61):
            P.line(st, R @ [k * 0.25 - 15, -15], R @ [k * 0.25 + 15, 15], rng.uniform(0.05, 0.1))
        m = np.maximum(m, st * inner)
        marks.append((worn(rng, P, m, rng.uniform(0, 0.6)), paint_tone()))
    for _ in range(rng.poisson(0.5)):  # arrows
        m = P.mask()
        c, s, a = rng.uniform(-3, 3, 2), rng.uniform(0.6, 1.5), ax + rng.choice([0, math.pi / 2, math.pi, -math.pi / 2])
        arrow = np.array([[-0.5, -0.08], [0.15, -0.08], [0.15, -0.22], [0.5, 0], [0.15, 0.22], [0.15, 0.08],
                          [-0.5, 0.08]]) * s
        P.poly(m, c + arrow @ rot2(a).T)
        marks.append((worn(rng, P, m, rng.uniform(0, 0.5)), paint_tone()))
    for _ in range(rng.poisson(0.4)):  # zone codes
        m = P.mask()
        code = f"{chr(65 + int(rng.integers(26)))}{int(rng.integers(1, 40))}"
        P.text(m, code, rng.uniform(-3, 3, 2), rng.uniform(0.3, 0.8), ax + rng.choice([0, math.pi / 2, math.pi]))
        marks.append((worn(rng, P, m, rng.uniform(0, 0.5)), paint_tone()))
    for m, tone in marks:
        t = t * (1 - m) + (tone + P.noise(0.2) * 6) * m
        paint = np.maximum(paint, m)

    # ---- dock levellers in front of the doors, the space beyond the wall
    for (x0, x1, yw, depth) in layout.get("levellers", []):
        lev = world_rect(*(R @ [(x0 + x1) / 2, yw - depth / 2]), x1 - x0, depth, ax)
        m = P.mask()
        P.poly(m, lev)
        tone = rng.uniform(95, 165)
        plate = np.full((h, w), tone, np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        pitch = max(3, P.L(0.03))
        cheq = ((xx // pitch + yy // pitch) % 2).astype(np.float32)  # alternating raised teardrops
        plate += (cheq * 2 - 1) * rng.uniform(6, 15) + P.noise(0.5) * 10
        t = t * (1 - m) + plate * m
        edge = P.mask()
        P.polyline(edge, lev, 0.02, closed=True)
        t += edge * rng.uniform(-60, -30)
    if layout.get("wall") is not None:
        yw, beyond = layout["wall"]
        m = P.mask()
        P.poly(m, (R @ np.array([[-40, yw], [40, yw], [40, yw + 40], [-40, yw + 40]]).T).T)
        if beyond == "dark":
            out = np.full((h, w), rng.uniform(4, 25), np.float32)
        elif beyond == "trailer":
            out = np.full((h, w), rng.uniform(60, 120), np.float32) + P.noise(0.15) * 15
            for k in range(-200, 200):  # floor planks of the trailer, along the dock direction
                pl = P.mask()
                P.line(pl, R @ [k * 0.12, yw], R @ [k * 0.12, yw + 20], 0.006)
                out += pl * -25
        else:  # bright yard
            out = np.full((h, w), rng.uniform(140, 230), np.float32) + P.noise(0.4) * 15
        t = t * (1 - m) + out * m

    if layout.get("mats"):
        t = paint_mats(t, P, layout["mats"], layout["mat_seed"])
    alb = np.clip(t / ALB, 0.0, 1.1).astype(np.float32)
    return alb, rough, {"floor": kind, "paint_share": float((paint > 0.5).mean())}


MAT_STYLES = {"rubber": 0.45, "scale": 0.25, "rug": 0.20, "plate": 0.10}


def paint_mats(t, P, mats, seed):
    """Floor mats / floor-scale platforms / rugs / cover plates (never labelled) painted into the floor texture.
    mats: [(cx, cy, l, w, yaw)] in world metres. Own RNG, so the rest of the floor is the same with or without."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:P.h, 0:P.w].astype(np.float32)
    c = P.cam
    X = (((xx + 0.5) / P.ss - 0.5) - c.cx) * c.mount_h / c.fx  # texture pixel -> floor plane (world metres)
    Y = (((yy + 0.5) / P.ss - 0.5) - c.cy) * c.mount_h / c.fy
    for cx, cy, l, w, yaw in mats:
        names = list(MAT_STYLES)
        style = names[rng.choice(len(names), p=np.array(list(MAT_STYLES.values())))]
        ca, sa = math.cos(yaw), math.sin(yaw)
        lx, ly = ca * (X - cx) + sa * (Y - cy), -sa * (X - cx) + ca * (Y - cy)  # mat frame
        rect = world_rect(cx, cy, l, w, yaw)
        if rng.random() < 0.45:  # frame / pit edge around it: steel angle or a lighter / darker concrete band
            fw = rng.uniform(0.03, 0.15)
            ring = P.mask()
            P.poly(ring, world_rect(cx, cy, l + 2 * fw, w + 2 * fw, yaw))
            t = t + ring * rng.choice([-1, 1]) * rng.uniform(10, 45)
        m = P.mask()
        P.poly(m, rect)
        if style == "rubber":       # black / dark grey rubber: bumps, ribs or coin pattern
            tone = rng.uniform(6, 40)
            p = rng.uniform(0.015, 0.05)
            pat = rng.choice(["bumps", "ribs", "plain"], p=[0.45, 0.35, 0.2])
            tex = np.full_like(t, tone)
            if pat == "bumps":
                tex += (np.sin(2 * math.pi * lx / p) * np.sin(2 * math.pi * ly / p) > 0.55) * rng.uniform(4, 14)
            elif pat == "ribs":
                tex += (np.sin(2 * math.pi * lx / p) > 0.3) * rng.uniform(4, 12)
        elif style == "scale":      # floor-scale platform: checker plate with teardrops, sometimes painted
            tone = rng.uniform(15, 80)
            p = rng.uniform(0.025, 0.05)
            tex = np.full_like(t, tone) + (((lx // p + ly // p) % 2) * 2 - 1) * rng.uniform(3, 12)
        elif style == "rug":        # textile / coir mat: fibrous noise, often with a border band
            tone = rng.uniform(15, 70)
            tex = np.full_like(t, tone) + P.noise(0.01) * rng.uniform(5, 15) + P.noise(0.1) * rng.uniform(3, 10)
        else:                       # steel cover plate
            tone = rng.uniform(70, 150)
            tex = np.full_like(t, tone) + P.noise(0.3) * rng.uniform(5, 15)
        tex += P.noise(rng.uniform(0.2, 1.0)) * rng.uniform(2, 10)   # dirt / wear
        if rng.random() < 0.5:  # border band inside the edge
            b = rng.uniform(0.02, 0.12)
            band = (np.abs(lx) > l / 2 - b) | (np.abs(ly) > w / 2 - b)
            tex += band * rng.choice([-1, 1]) * rng.uniform(5, 30)
        t = t * (1 - m) + tex * m
        edge = P.mask()
        P.polyline(edge, rect, rng.uniform(0.004, 0.015), closed=True)
        t += edge * rng.uniform(-40, -10)
    return t


# ---------------------------------------------------------------------------------- structures
def wall_with_doors(rng, cam, ax, yw, top):
    """Wall along the layout x axis at layout y = yw (the far side y > yw is outside), dock doors in it.
    -> (things, levellers [(x0, x1, yw, depth)], door centres (layout x))."""
    tw = rng.uniform(0.2, 0.35)
    wall_tone = rng.uniform(100, 200) if rng.random() < 0.7 else rng.uniform(50, 100)
    panel = rng.random() < 0.5

    def wtex(f, w, h):
        t = G.plain(rng, w, h, wall_tone, 3, 6)
        if panel:  # sandwich panel seams
            for y in range(0, h, max(4, G.px(1.0))):
                t[y:y + 2] -= 25
        return t
    things, levellers, doors = [], [], []
    pitch = rng.uniform(3.6, 4.6)
    off = rng.uniform(0, pitch)
    dw, dh = rng.uniform(2.5, 3.1), rng.uniform(2.8, 3.6)
    xs = [off + k * pitch for k in range(-3, 4)]
    parts = []
    edges = [-15.0]
    for x in xs:
        edges += [x - dw / 2, x + dw / 2]
    edges.append(15.0)
    for a, b in zip(edges[0::2], edges[1::2]):  # wall segments between the doors
        parts.append(box_p((a + b) / 2, yw + tw / 2, b - a, tw, 0.0, top, wtex, "matte"))
    seal_tone = rng.uniform(12, 35)
    door_tone = rng.uniform(90, 190)
    for x in xs:
        parts.append(box_p(x, yw + tw / 2, dw, tw, dh, top, wtex, "matte"))  # lintel
        state = rng.choice(["open", "half", "closed"], p=[0.55, 0.25, 0.2])
        if state != "open":  # sectional door: ribbed panel in the wall plane
            z0 = 0.0 if state == "closed" else rng.uniform(1.0, dh - 0.3)
            ribs = lambda f, w, h: G.plain(rng, w, h, door_tone, 2, 4) - \
                ((np.arange(h)[:, None] % max(4, G.px(0.5))) < 3) * 30  # noqa: E731
            parts.append(box_p(x, yw + tw * 0.7, dw, 0.05, z0, dh, ribs, "paint"))
        for s in (-1, 1):  # dock shelter / seals, door guards
            parts.append(box_p(x + s * (dw / 2 + 0.08), yw - 0.15, 0.16, 0.3, 0.0, dh + 0.1,
                               tex_plain(rng, seal_tone, 2, 3), "rubber"))
            if rng.random() < 0.7:
                parts.append(cylinder(x + s * (dw / 2 + 0.3), yw - 0.45, rng.uniform(0.06, 0.1), 0.0,
                                      rng.uniform(0.9, 1.3), tex_bands(rng, rng.uniform(130, 175), 20), "paint"))
        if state != "closed" and rng.random() < 0.85:
            levellers.append((x - dw / 2 + 0.15, x + dw / 2 - 0.15, yw, rng.uniform(2.0, 3.0)))
        doors.append(x)
    things.append(Thing(parts, yaw=ax, foot=(rot2(ax) @ np.array([[-30, yw - 0.6], [30, yw - 0.6], [30, yw + 30],
                                                                   [-30, yw + 30]]).T).T))
    return things, levellers, doors


def column(rng, x, y, top, yaw):
    s = rng.uniform(0.35, 0.6)
    parts = []
    if rng.random() < 0.5:  # concrete
        parts.append(box_p(0, 0, s, s, 0, top, tex_plain(rng, rng.uniform(110, 190), 3, 8), "matte"))
    else:  # steel H profile
        fl, tw_ = rng.uniform(0.25, 0.4), rng.uniform(0.012, 0.025)
        tone = rng.uniform(60, 170)
        for sy in (-1, 1):
            parts.append(box_p(0, sy * (fl / 2 - tw_ / 2), fl, tw_, 0, top, tex_plain(rng, tone, 2, 4), "paint"))
        parts.append(box_p(0, 0, tw_, fl, 0, top, tex_plain(rng, tone, 2, 4), "paint"))
        s = fl
    if rng.random() < 0.6:  # impact guard
        g = s + rng.uniform(0.1, 0.25)
        gh = rng.uniform(0.5, 1.2)
        tex = tex_hazard(rng) if rng.random() < 0.6 else tex_plain(rng, rng.uniform(120, 170))
        for k in range(4):
            a = k * math.pi / 2
            c = rot2(a) @ [g / 2, 0]
            parts.append(box_p(c[0], c[1], 0.03, g + 0.03, 0, gh, tex, "paint", yaw=-a))
        s = g
    return Thing(parts, yaw=yaw, cx=x, cy=y, foot=world_rect(x, y, s + 0.1, s + 0.1, yaw))


def guard_rail(rng, a, b):
    """Posts + 1-2 horizontal rails from a to b (world xy)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = np.linalg.norm(b - a)
    d = (b - a) / max(L, 1e-9)
    bright = rng.random() < 0.4
    tone = rng.uniform(150, 200) if bright else rng.uniform(110, 170)
    mat = "steel" if bright else "paint"
    tex = tex_plain(rng, tone, 2, 3) if bright or rng.random() < 0.6 else tex_hazard(rng)
    hs = sorted(rng.uniform(0.25, 1.1, int(rng.integers(1, 3))))
    r = rng.uniform(0.03, 0.06)
    parts = []
    for s in np.arange(0, L + 1e-6, rng.uniform(1.2, 2.2)):
        p = a + d * s
        parts.append(cylinder(p[0], p[1], r * 1.2, 0.0, hs[-1] + 0.05, tex, mat, n=12))
    for hz in hs:
        if rng.random() < 0.5:
            parts.append(U.tube([np.r_[a, hz], np.r_[(a + b) / 2, hz], np.r_[b, hz]], r, tex, mat, n=10))
        else:
            parts.append(beam(np.r_[a, hz], np.r_[b, hz], r * 2, rng.uniform(0.08, 0.3), tex, mat))
    yaw = math.atan2(d[1], d[0])
    c = (a + b) / 2
    return Thing(parts, foot=world_rect(c[0], c[1], L + 0.2, 0.3, yaw))


def rack(rng, cam, a, b, depth, side):
    """Pallet rack along a -> b (world xy), `depth` deep towards `side` (+1/-1 normal), with loads on the beams."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    n = np.array([-d[1], d[0]]) * side
    up_tone, beam_tone = rng.uniform(40, 120), rng.uniform(80, 160)
    utex, btex = tex_plain(rng, up_tone, 2, 3), tex_plain(rng, beam_tone, 2, 3)
    top = min(cam.mount_h - 0.8, rng.uniform(3.5, 9.0))
    levels = [z for z in np.arange(rng.uniform(1.2, 1.8), top, rng.uniform(1.3, 1.9))]
    bay = rng.uniform(2.6, 2.9)
    parts = []
    xs = np.arange(0, L + 1e-6, bay)
    for s in xs:
        for k in (0.0, depth):
            p = a + d * s + n * k
            parts.append(G.part(G.prism(p[0], p[1], 0.1, 0.08, 0.0, top, math.atan2(d[1], d[0])), utex, mat="paint"))
        p0, p1 = a + d * s, a + d * s + n * depth
        for z in np.arange(0.3, top, 0.6):  # bracing
            parts.append(beam(np.r_[p0, z], np.r_[p1, z + 0.5], 0.03, 0.03, utex, "paint"))
    for z in levels:
        for k in (0.0, depth):
            p0, p1 = a + n * k, b + n * k
            parts.append(beam(np.r_[p0, z], np.r_[p1, z], 0.05, 0.12, btex, "paint"))
    for z in levels[-2:]:  # loads on the upper levels (seen from above), not on the floor -> never labelled
        for s in xs[:-1]:
            for j in range(2):
                if rng.random() < 0.3:
                    continue
                if top - z - 0.1 < 0.5:
                    continue
                c = a + d * (s + bay * (0.27 + 0.46 * j)) + n * depth / 2
                pl = G.build_pallet(rng, 1.2, 0.8)
                H = rng.uniform(0.4, min(1.3, top - z - 0.1))
                lp = G.load_stack(rng, 1.15, 0.78, H, G.PALLET_H, [G.make_sku(rng)])
                U.xform(pl + lp, U.rotz(math.atan2(n[1], n[0]) + rng.normal(0, 0.02)), (c[0], c[1], z + 0.06))
                parts += pl + lp
    c = a + d * L / 2 + n * depth / 2
    yaw = math.atan2(d[1], d[0])
    return Thing(parts, foot=world_rect(c[0], c[1], L + 0.3, depth + 0.3, yaw))


def conveyor(rng, a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    L = np.linalg.norm(b - a)
    d = (b - a) / L
    yaw = math.atan2(d[1], d[0])
    W, z = rng.uniform(0.5, 1.0), rng.uniform(0.5, 0.9)
    tone = rng.uniform(60, 160)
    ftex = tex_plain(rng, tone, 2, 3)
    parts = []
    for s in (-1, 1):  # side frames
        parts.append(box_p(L / 2, s * W / 2, L, 0.05, z - 0.12, z + 0.03, ftex, "paint"))
    for x in np.arange(0.3, L, 1.5):  # legs
        for s in (-1, 1):
            parts.append(box_p(x, s * (W / 2 - 0.03), 0.05, 0.05, 0.0, z - 0.12, ftex, "paint"))
    if rng.random() < 0.6:  # rollers
        rtex = tex_plain(rng, rng.uniform(140, 200), 2, 2)
        r = rng.uniform(0.02, 0.03)
        for x in np.arange(0.05, L, rng.uniform(0.07, 0.12)):
            parts.append(U.tube([[x, -W / 2 + 0.03, z - r], [x, W / 2 - 0.03, z - r]], r, rtex, "steel", n=8))
    else:  # belt
        parts.append(box_p(L / 2, 0, L, W - 0.08, z - 0.03, z, tex_plain(rng, rng.uniform(15, 50), 2, 3), "rubber"))
    for x in np.arange(rng.uniform(0.2, 1.5), L - 0.4, rng.uniform(0.8, 2.5)):  # parcels riding on it
        if rng.random() < 0.35:
            l, w_, hh = rng.uniform(0.2, 0.5), rng.uniform(0.15, min(0.45, W - 0.1)), rng.uniform(0.1, 0.4)
            parts.append(G.carton_part(rng, G.make_sku(rng), x, rng.uniform(-0.05, 0.05), l, w_, z, z + hh,
                                       rng.normal(0, 0.1), cull=False))
    return Thing(parts, yaw=yaw, cx=a[0], cy=a[1],
                 foot=world_rect(*(a + d * L / 2), L + 0.2, W + 0.2, yaw))


def forklift(rng):
    """Counterbalance forklift, local frame: forks along +x, footprint ~ x -1.3 .. 2.1, y +-0.6."""
    paint = rng.uniform(80, 180)
    dark = rng.uniform(15, 45)
    pt, dt = tex_plain(rng, paint, 3, 5), tex_plain(rng, dark, 2, 3)
    W = rng.uniform(1.0, 1.2)
    parts = U.rbox((1.5, W, 0.9), pt, "paint", e1=0.15, e2=0.15, n=8)
    U.xform(parts, t=(0.0, 0.0, 0.6))
    cw = U.rbox((0.55, W, 1.0), dt if rng.random() < 0.5 else pt, "paint", e1=0.3, e2=0.5, n=8)
    parts += U.xform(cw, t=(-0.95, 0.0, 0.62))
    for x in (-0.75, 0.55):  # wheels
        for s in (-1, 1):
            prof = [(0.08, -0.09), (0.22, -0.09), (0.25, -0.06), (0.25, 0.06), (0.22, 0.09), (0.08, 0.09),
                    (0.08, -0.09)]
            parts.append(U.lathe_at(prof, dt, "rubber", R=U.rotx(math.pi / 2), t=(x, s * (W / 2 - 0.05), 0.25), n=16))
    roof = rng.uniform(2.0, 2.25)
    ptex = tex_plain(rng, dark if rng.random() < 0.6 else paint, 2, 3)
    for x in (0.45, -0.75):  # overhead guard posts
        for s in (-1, 1):
            parts.append(beam([x, s * (W / 2 - 0.06), 1.0], [x - 0.1, s * (W / 2 - 0.06), roof], 0.06, 0.06, ptex,
                              "paint"))
    for s in (-1, 1):
        parts.append(beam([0.38, s * (W / 2 - 0.06), roof], [-0.88, s * (W / 2 - 0.06), roof], 0.07, 0.05, ptex,
                          "paint"))
    for x in np.arange(-0.8, 0.35, 0.12):  # roof slats
        parts.append(beam([x, -W / 2 + 0.06, roof], [x, W / 2 - 0.06, roof], 0.04, 0.03, ptex, "paint"))
    seat = U.rbox((0.45, 0.45, 0.35), dt, "rubber", e1=0.4, e2=0.4, n=6)
    parts += U.xform(seat, t=(-0.35, 0.0, 1.2))
    parts.append(U.tube(np.c_[U.circle(0.15, 0.0, 0.17, 16), np.full(16, 1.45)], 0.015, dt, "rubber", closed=True))
    mh = rng.uniform(2.0, 2.4)  # mast + carriage + forks
    mt = tex_plain(rng, rng.uniform(25, 80), 2, 3)
    for s in (-1, 1):
        parts.append(box_p(0.85, s * 0.33, 0.1, 0.1, 0.1, mh, mt, "paint"))
    for z in (0.5, mh - 0.1):
        parts.append(box_p(0.85, 0, 0.08, 0.76, z, z + 0.08, mt, "paint"))
    lift = rng.uniform(0.02, 0.35)
    parts.append(box_p(0.95, 0, 0.05, 1.0, lift, lift + 0.5, mt, "paint"))
    flen = rng.uniform(1.0, 1.2)
    for s in (-1, 1):
        y = s * rng.uniform(0.2, 0.35)
        parts.append(box_p(0.97 + flen / 2, y, flen, 0.12, lift, lift + 0.045, mt, "steel"))
    if rng.random() < 0.3:  # empty pallet on the forks
        parts += U.xform(G.build_pallet(rng, 1.2, 0.8), U.rotz(math.pi / 2 * rng.integers(2)),
                         (0.97 + 0.6, 0.0, lift + 0.045))
    return parts, (-1.3, 0.97 + flen + 0.1, W + 0.2)


def pallet_jack(rng):
    """Electric pallet truck / stacker: hand pallet truck + battery body + tiller."""
    parts = U.build_truck(rng)
    body = U.rbox((0.55, 0.78, rng.uniform(0.9, 1.5)), tex_plain(rng, rng.uniform(60, 170), 3, 5), "paint",
                  e1=0.2, e2=0.3, n=6)
    hgt = body[0]["surf"][..., 2].max() - body[1]["surf"][..., 2].min()
    parts += U.xform(body, t=(-0.45, 0.0, hgt / 2 + 0.05))
    return parts, (-0.8, 1.25, 0.85)


def person(rng):
    """Standing / walking person, local frame facing +x; ~1.6-1.9 m."""
    k = rng.uniform(0.9, 1.08)
    trousers = rng.uniform(20, 90)
    vest = rng.random() < 0.6
    jacket = rng.uniform(140, 190) if vest else rng.uniform(20, 120)
    stripe = rng.uniform(200, 235)

    def torso_tex(f, w, h):
        t = G.plain(rng, w, h, jacket, 3, 6)
        if vest:
            for y in (int(h * 0.35), int(h * 0.65)):
                t[y:y + max(2, h // 12)] = stripe
            t[:, w // 2 - max(1, w // 20):w // 2 + max(1, w // 20)] = stripe
        return t
    parts = []
    step = rng.uniform(-0.25, 0.25) if rng.random() < 0.6 else 0.0
    for s in (-1, 1):  # legs
        leg = U.rbox((0.15 * k, 0.15 * k, 0.85 * k), tex_plain(rng, trousers, 2, 4), "fabric", e1=0.5, e2=0.8, n=6)
        parts += U.xform(leg, t=(s * step / 2, s * 0.1 * k, 0.43 * k))
    torso = U.rbox((0.26 * k, 0.44 * k, 0.62 * k), torso_tex, "fabric", e1=0.5, e2=0.6, n=8)
    parts += U.xform(torso, t=(0.0, 0.0, 1.16 * k))
    carry = rng.random() < 0.25
    for s in (-1, 1):  # arms
        hand = [0.3 * k, s * 0.15 * k, 1.05 * k] if carry else [-s * step * 0.6, s * 0.27 * k, 0.8 * k]
        parts.append(U.tube([[0.0, s * 0.24 * k, 1.42 * k], hand], 0.05 * k, tex_plain(rng, jacket, 3, 4), "fabric",
                            n=8))
    head_tone = rng.uniform(15, 60) if rng.random() < 0.6 else rng.uniform(150, 220)  # hair / helmet
    parts.append(sphere(0.0, 0.0, 1.66 * k, 0.11 * k, tex_plain(rng, head_tone, 2, 4), "matte", flat=0.1))
    if carry:
        parts.append(G.carton_part(rng, G.make_sku(rng), 0.38 * k, 0, rng.uniform(0.25, 0.5), rng.uniform(0.2, 0.4),
                                   0.95 * k, 0.95 * k + rng.uniform(0.15, 0.35), 0.0, cull=False))
    return parts, 0.4


def empty_pallets(rng):
    L, W = [(1.2, 0.8), (1.2, 1.0), (0.8, 0.6), (1.2, 0.8)][int(rng.integers(4))]
    n = 1 if rng.random() < 0.45 else int(rng.integers(2, 16))
    parts = []
    for i in range(n):
        p = G.build_pallet(rng, L, W)
        parts += U.xform(p, U.rotz(rng.normal(0, 0.03)), (rng.normal(0, 0.015), rng.normal(0, 0.015),
                                                          i * G.PALLET_H))
    return parts, (L + 0.06, W + 0.06)


def misc_item(rng):
    """-> (parts, footprint (l, w)) of a small floor object."""
    r = rng.random()
    if r < 0.15:  # waste bin
        rr, hh = rng.uniform(0.25, 0.4), rng.uniform(0.6, 1.1)
        prof = [(0.0, 0.0), (rr, 0.0), (rr, hh), (rr - 0.015, hh), (rr - 0.015, 0.03), (0.0, 0.03)]
        return [U.lathe_at(prof, tex_plain(rng, rng.uniform(20, 160), 2, 4), "plastic", n=20)], (2 * rr, 2 * rr)
    if r < 0.27:  # cone
        h = rng.uniform(0.45, 0.75)
        prof = [(0.0, 0.0), (0.2, 0.0), (0.2, 0.03), (0.04, h), (0.0, h)]
        return [U.lathe_at(prof, tex_bands(rng, rng.uniform(90, 140), rng.uniform(190, 230), 2), "plastic", n=16)], \
            (0.42, 0.42)
    if r < 0.4:  # bollard
        rr = rng.uniform(0.05, 0.11)
        return [cylinder(0, 0, rr, 0.0, rng.uniform(0.8, 1.2), tex_bands(rng, rng.uniform(130, 175), 20), "paint")], \
            (2 * rr, 2 * rr)
    if r < 0.55:  # cable / hose on the floor
        rr = rng.uniform(0.008, 0.025)
        pts = [np.zeros(2)]
        a = rng.uniform(0, 2 * math.pi)
        for _ in range(12):
            a += rng.normal(0, 0.5)
            pts.append(pts[-1] + rng.uniform(0.15, 0.4) * np.array([math.cos(a), math.sin(a)]))
        pts = np.array(pts) - np.mean(pts, 0)
        path = np.c_[U.chaikin(pts, 2)[:-3], np.full(len(U.chaikin(pts, 2)) - 3, rr)]
        tone = rng.uniform(15, 50) if rng.random() < 0.6 else rng.uniform(100, 170)
        ext = np.ptp(pts, 0) + 0.1
        return [U.tube(path, rr, tex_plain(rng, tone, 2, 2), "rubber", n=8)], tuple(ext)
    if r < 0.68:  # cardboard sheet / slip sheet
        l, w = rng.uniform(0.5, 1.3), rng.uniform(0.4, 1.0)
        return [box_p(0, 0, l, w, 0, rng.uniform(0.003, 0.01), tex_plain(rng, rng.uniform(100, 170), 3, 6),
                      "cardboard")], (l, w)
    if r < 0.76:  # stretch film roll lying
        ln = rng.uniform(0.45, 0.5)
        prof = [(0.0, -ln / 2), (0.06, -ln / 2), (0.06, ln / 2), (0.0, ln / 2)]
        return [U.lathe_at(prof, tex_plain(rng, rng.uniform(170, 220), 2, 3), "glossy", R=U.rotx(math.pi / 2),
                           t=(0, 0, 0.06), n=16)], (0.15, ln)
    if r < 0.9:  # platform trolley
        l, w = rng.uniform(0.8, 1.2), rng.uniform(0.5, 0.8)
        tone = rng.uniform(40, 160)
        parts = [box_p(0, 0, l, w, 0.14, 0.17, tex_plain(rng, tone, 2, 4), "paint")]
        for sx in (-1, 1):
            for sy in (-1, 1):
                parts += U.castor(rng, sx * (l / 2 - 0.08), sy * (w / 2 - 0.06))
        hz = rng.uniform(0.8, 1.0)
        parts.append(U.tube([[-l / 2 + 0.02, -w / 2 + 0.04, 0.17], [-l / 2 + 0.02, -w / 2 + 0.04, hz],
                             [-l / 2 + 0.02, w / 2 - 0.04, hz], [-l / 2 + 0.02, w / 2 - 0.04, 0.17]], 0.015,
                            tex_plain(rng, rng.uniform(30, 180), 2, 2), "steel", n=8))
        return parts, (l + 0.1, w + 0.1)
    # desk / packing station with a screen
    l, w, hz = rng.uniform(1.2, 2.0), rng.uniform(0.6, 0.9), rng.uniform(0.75, 0.95)
    tone = rng.uniform(80, 200)
    parts = [box_p(0, 0, l, w, hz - 0.03, hz, tex_plain(rng, tone, 2, 4), "paint")]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(box_p(sx * (l / 2 - 0.05), sy * (w / 2 - 0.05), 0.04, 0.04, 0, hz - 0.03,
                               tex_plain(rng, 60, 2, 2), "steel"))
    parts.append(box_p(rng.uniform(-l / 4, l / 4), -w / 4, 0.5, 0.06, hz, hz + 0.35, tex_plain(rng, 20, 2, 2),
                       "plastic"))
    return parts, (l + 0.05, w + 0.05)


def overhead_pipe(rng, cam):
    """Pipe / cable tray crossing between camera and floor (seen big and soft)."""
    z = cam.mount_h - rng.uniform(0.5, 1.5)
    a = rng.uniform(0, math.pi)
    d = np.array([math.cos(a), math.sin(a)])
    c = rng.uniform(-1.5, 1.5, 2)
    p0, p1 = c - d * 10, c + d * 10
    tone = rng.uniform(60, 200)
    if rng.random() < 0.5:
        return Thing([U.tube([np.r_[p0, z], np.r_[p1, z]], rng.uniform(0.04, 0.15), tex_plain(rng, tone, 2, 3),
                             "steel", n=12)])
    return Thing([beam(np.r_[p0, z], np.r_[p1, z], rng.uniform(0.2, 0.5), 0.08, tex_plain(rng, tone, 3, 5), "steel")])
