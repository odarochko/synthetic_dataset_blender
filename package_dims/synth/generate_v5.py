"""Synthetic dataset v5: diverse cameras, floors, surroundings, scene layouts and units, path-traced with Blender
Cycles (synth/blender_render_v5.py). Same output schema / framing / label convention as v1-v4, so it trains with
the same custom_3d_model loader (intrinsics are read per image).

v4 put 0-2 isolated units on ONE real floor image seen by ONE camera. v5 varies everything per image and takes
nothing from cam_000CDF0AD90F (its 19 frames stay an independent test set: no real floor image, not its camera
model, no tone / noise / sun-patch matching to its frames, label size not fixed to the size fitted on them):

  camera     JAI GO-5000M + KOWA LM8HC / LM12HC / LM16HC, factory calibrations of the shipped models + jitter,
             mounted 2.6-8.0 m (env_v5.make_camera); black undistortion corners as in the real framing
  floor      procedural and rendered (lit, glossy, shadowed): concrete / worn / polished / epoxy / dark painted /
             asphalt / tiles with joints, cracks, stains, tyre marks, markings (lanes, bays, hatching, arrows,
             codes), dock levellers, the bay / trailer / yard beyond a dock wall (env_v5.make_floor)
  surroundings  walls with dock doors, columns, guard rails, pallet racks, conveyors, forklifts, electric pallet
             trucks, people, empty pallets / stacks, bins, cones, bollards, cables, sheets, trolleys, desks,
             overhead pipes -- never labelled
  layouts    isolated units | rows along the floor axis (touching / tight / spaced, units cut by the image edge) |
             random clusters | dock scenes | loose parcels | empty scenes; up to 10 labelled units per image
  units      v4 units (v2 cartons / pallet loads / cages upgraded + units_v3 kinds) + double-stacked pallet loads,
             half / quarter pallets, loose parcels; random size scaling and tone (dark / bright goods)
  light      hall lamps, night (low ambient, lamp pools), daylight (skylights), sun beams, light through the doors
  camera fx  auto exposure, gain, gamma, vignetting, blur, shot + read noise

Labels: units fully inside the frame (all 8 box corners inside the valid image) are labelled; units cut by the
image edge are only placed when their centre is outside the frame and are not labelled.

    nice -n 10 python3 synth/generate_v5.py --n 3000 --workers 6 --seed 0   # -> datasets/SYNTH_V5_DIVERSE
    python3 synth/generate_v5.py --n 24 --out /tmp/v5_preview --seed 5       # quick look
Interrupted runs resume: images whose png + json already exist are skipped.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import env_v5 as E  # noqa: E402
import generate as GEN  # noqa: E402
import generate_v2 as G  # noqa: E402
import generate_v3 as V3  # noqa: E402
import units_v3 as U  # noqa: E402
from camera import pad_square  # noqa: E402
from generate import SHARD, rect_corners  # noqa: E402

ALB, EXPOSURE = V3.ALB, V3.EXPOSURE
V6 = False         # --v6: floor mats / scales / rugs, more cages and overhanging loads (set in main before the pool)
V6_CAGE_P, V6_OVERHANG_P, V6_MAT_P = 0.10, 0.10, 0.5
V7 = False         # --v7 (implies --v6): damaged loads, torn film flaps, labels near edges / on cages / damaged
V7_DAMAGE_P, V7_FLAP_P, V7_EDGE_LABEL_P, V7_LABEL_DAMAGE_P = 0.8, 0.5, 0.5, 0.5
MAX_LABELLED = 10  # custom_3d_model max_parcels
MODES = {"isolated": 0.17, "rows": 0.30, "cluster": 0.22, "dock": 0.12, "parcels": 0.09, "empty": 0.10}
ROUND = {"drums", "cable_drum", "coils", "tyres", "rolls_standing", "rolls_lying", "tubes", "buckets", "ibc"}


# ---------------------------------------------------------------------------------- units
def set_scales(rng, cam):
    """Texture resolution for this camera (px per metre at the top of a 2 m unit) and the label size."""
    ppm = float(np.clip(cam.fx / max(cam.mount_h - 2.0, 1.0), 250, 420))
    G.TEX_PPM = GEN.TEX_PPM = ppm
    G.LABEL_M = GEN.LABEL_M = float(rng.uniform(0.075, 0.115))


def shift_unit(u, R, t):
    """Rigid transform of a whole unit (parts, mesh quads, films, label spots) in place."""
    U.xform(u["parts"], R, t)
    u["quads"] = [(q[0], q[1] @ R.T + t) + tuple(q[2:]) for q in u["quads"]]
    for f in u["films"]:
        f["P"] = f["P"] @ R.T + t
        f["N"] = f["N"] @ R.T
    if u["labels"]:
        u["labels"] = [tuple((np.array(s[:3]) @ R.T + t).tolist()) + (s[3],) for s in u["labels"]]
    return u


def double_stack(rng, hmax):
    """Two pallet loads stacked (common in cross-docks); the label box encloses both."""
    def load(h0, h1):
        for _ in range(30):
            u = V3.v2_unit(rng)
            if u["pallet"] and u["kind"] in ("stack", "mixed", "single") and h0 <= V3.unit_extent(u)[2] <= h1:
                return u
        return None
    a = load(0.5, min(1.4, hmax - 0.5))
    if a is None:
        return None
    Ha = V3.unit_extent(a)[2]
    b = load(0.35, min(1.3, hmax - Ha))
    if b is None:
        return None
    shift_unit(b, U.rotz(rng.normal(0, 0.03) + (math.pi if rng.random() < 0.5 else 0.0)),
               np.array([rng.normal(0, 0.03), rng.normal(0, 0.03), Ha + 0.002]))
    parts = a["parts"] + b["parts"]
    H = V3.unit_extent({"parts": parts, "quads": a["quads"] + b["quads"], "films": a["films"] + b["films"]})[2]
    top = [p for p in b["parts"] if p["P"][:, 2].max() > H - 0.05]
    u = dict(a, kind="double_stack", parts=parts, quads=a["quads"] + b["quads"], films=a["films"] + b["films"],
             labels=b["labels"] if b["labels"] else U.top_label_spots(rng, top, H))
    return u


def small_pallet(rng):
    """Half (0.8 x 0.6) / quarter (0.6 x 0.4) pallet with cartons."""
    L, W = [(0.8, 0.6), (0.8, 0.6), (0.6, 0.4)][int(rng.integers(3))]
    parts = U.carrier(G.build_pallet(rng, L, W))
    H = rng.uniform(0.25, 1.4)
    skus = [G.make_sku(rng)] if rng.random() < 0.7 else [G.make_sku(rng) for _ in range(2)]
    lp = G.load_stack(rng, L + rng.uniform(-0.08, 0.04), W + rng.uniform(-0.08, 0.04), H, G.PALLET_H, skus)
    films = U.maybe_film(rng, lp, G.PALLET_H, p=0.45)
    return U.unit("small_pallet", parts + lp, pallet=(L, W), films=films,
                  labels=U.top_label_spots(rng, lp, G.PALLET_H + H))


def parcel(rng, big=False):
    """Loose carton on the floor (no pallet)."""
    sku = G.make_sku(rng)
    hi = 1.0 if big else 0.7
    l, w = sorted([rng.uniform(0.2, hi), rng.uniform(0.15, hi * 0.8)], reverse=True)
    H = rng.uniform(0.08, 0.8 if big else 0.6)
    parts = [G.carton_part(rng, sku, 0, 0, l, w, 0, H, 0.0, cull=False)]
    if rng.random() < 0.3:
        parts = U.soften(rng, parts)
    films = U.maybe_film(rng, parts, 0.0, p=0.08)
    return U.unit("parcel", parts, films=films, labels=U.top_label_spots(rng, parts, H), typ="parcel")


def scale_unit(u, sx, sy, sz):
    """Anisotropic size change: footprint x / y, load height above the carrier."""
    c = u["carrier_h"]
    S = np.array([sx, sy, sz])

    def P_(P):
        P = np.array(P, np.float64)
        P[..., 0] *= sx
        P[..., 1] *= sy
        z = P[..., 2]
        P[..., 2] = np.where(z > c, c + (z - c) * sz, z)
        return P

    def N_(N):
        N = np.asarray(N, np.float64) / S
        return N / np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
    for p in u["parts"]:
        if "surf" in p:
            p["surf"], p["N"] = P_(p["surf"]), N_(p["N"])
            p["P"] = p["surf"].reshape(-1, 3)
        elif "polys" in p:
            p["polys"] = [(P_(q), P_(np.asarray(ins)[None])[0]) for q, ins in p["polys"]]
            p["P"] = np.concatenate([q for q, _ in p["polys"]])
        else:
            p["P"] = P_(p["P"])
    u["quads"] = [(q[0], P_(q[1])) + tuple(q[2:]) for q in u["quads"]]
    for f in u["films"]:
        f["P"], f["N"] = P_(f["P"]), N_(f["N"])
    if u["labels"]:
        u["labels"] = [(x * sx, y * sy, float(P_(np.array([[0, 0, z]]))[0, 2]), ms * min(sx, sy))
                       for x, y, z, ms in u["labels"]]
    if u["pallet"]:
        u["pallet"] = [u["pallet"][0] * sx, u["pallet"][1] * sy]
    u["load"] = [u["load"][0] * sx, u["load"][1] * sy]


def retone(rng, u):
    """Darker / brighter goods: scale the tone of the load textures (not the pallet / carrier)."""
    k = float(np.clip(np.exp(rng.normal(0, 0.35)), 0.35, 1.6))
    add = rng.uniform(-10, 10)
    c = u["carrier_h"]
    for p in u["parts"]:
        if p.get("carrier") or p["P"][:, 2].min() < c - 1e-6 or p.get("mat") == "strap":
            continue
        fn = p["tex"]

        def g(f, w, h, fn=fn):
            t = fn(f, w, h)
            t = t[0] if isinstance(t, tuple) else t
            return np.clip(np.asarray(t, np.float32) * k + add, 0, 255)
        p["tex"] = g
        if p.get("top_tex") is not None:
            p["top_tex"] = np.clip(np.asarray(p["top_tex"], np.float32) * k + add, 0, 255)


def cage_unit(rng):
    """v6: gitterbox cage (v2) or roll container, resized well beyond the pallet sizes (real cages reach ~1.6 m)."""
    if rng.random() < 0.4:
        return U.make_unit(rng, kinds=["roll_container"])
    for _ in range(60):
        u = V3.v2_unit(rng)
        if u["kind"] == "cage":
            scale_unit(u, rng.uniform(0.9, 1.4), rng.uniform(0.9, 1.45), rng.uniform(0.9, 1.3))
            return u
    return None


def overhang_unit(rng):
    """v6: pallet load longer than its pallet (panel / board / big box overhanging the pallet ends)."""
    for _ in range(60):
        u = V3.v2_unit(rng)
        if u["pallet"] and u["kind"] != "cage" and u["load"][0] > u["pallet"][0] + 0.08:
            return u
    return None


# ---------------------------------------------------------------------------------- v7: damaged loads, labels
def unit_ext(u):
    """(lo, hi, H) of the package box. v7 fixes it before adding torn film flaps: they are not part of the size."""
    return u["ext"] if "ext" in u else V3.unit_extent(u)


def damage_load(rng, u):
    """v7: a knocked-about carton load -- cartons missing from the top layer, the upper layer(s) slid sideways and
    twisted, the whole load leaning. Only loads of plain box parts (v2 cartons); straps are dropped, the film is
    re-wrapped over the new outline, label spots are recomputed. -> list of what was done."""
    c = u["carrier_h"]
    load = [p for p in u["parts"] if not p.get("carrier") and p["P"][:, 2].min() >= c - 1e-6
            and p.get("mat") != "strap"]
    if u["kind"] in ROUND or u["kind"] == "cage" or len(load) < 2 or not all(U.is_box(p) for p in load):
        return []
    rest = [p for p in u["parts"] if not any(p is q for q in load) and p.get("mat") != "strap"]
    H = max(p["P"][:, 2].max() for p in load)
    done = []
    ops = {k for k, q in (("missing", 0.5), ("shifted", 0.5), ("leaning", 0.4)) if rng.random() < q}
    ops = ops or {("missing", "shifted", "leaning")[int(rng.integers(3))]}
    top = [i for i, p in enumerate(load) if p["P"][:, 2].max() > H - 0.03]
    if len(top) >= 3 and "missing" in ops:
        drop = set(rng.choice(top, size=int(rng.integers(1, len(top) // 2 + 1)), replace=False).tolist())
        load = [p for i, p in enumerate(load) if i not in drop]
        done.append("missing")
    bottoms = sorted({round(float(p["P"][:, 2].min()), 3) for p in load if p["P"][:, 2].min() > c + 0.05})
    if bottoms and "shifted" in ops:
        zc = bottoms[-1] if rng.random() < 0.6 else bottoms[int(rng.integers(len(bottoms)))]
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(0.02, 0.09) * np.array([math.cos(a), math.sin(a), 0.0])
        up = [p for p in load if p["P"][:, 2].min() >= zc - 1e-3]
        cen = np.concatenate([p["P"] for p in up]).mean(0) * [1, 1, 0]
        R = U.rotz(rng.normal(0, 0.04))  # twist about the layer centre, then slide
        U.xform(up, R, cen - cen @ R.T + d)
        done.append("shifted")
    if "leaning" in ops:
        a, k = rng.uniform(0, 2 * math.pi), math.tan(math.radians(rng.uniform(0.8, 3.0)))
        for p in load:
            p["P"] = p["P"] + np.c_[k * math.cos(a) * (p["P"][:, 2] - c), k * math.sin(a) * (p["P"][:, 2] - c),
                                    np.zeros(len(p["P"]))]
        done.append("leaning")
    if not done:
        return []
    for p in load:  # faces of slid / missing neighbours are now partly in view
        p["cull"] = False
    if u["films"]:
        f0 = u["films"][0]
        style = {k: f0[k] for k in ("style", "tone", "base", "gain")}
        z0 = float(min(f["P"][..., 2].min() for f in u["films"]))
        H2 = max(p["P"][:, 2].max() for p in load)
        has_top = any(f["kind"] == "top" for f in u["films"])
        u["films"] = U.film_wrap(rng, U.comps_of(load), z0, H2 if has_top else H2 - rng.uniform(0.0, 0.1),
                                 has_top, style=style)
    u["parts"] = rest + load
    if u["labels"] is not None:
        u["labels"] = U.top_label_spots(rng, load, max(p["P"][:, 2].max() for p in load))
    lo, hi = U.extent(load)
    u["load"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    return done


def film_flaps(rng, u):
    """v7: 1-3 torn stretch-film flaps peeling off the side film, hanging / curling outwards from their attachment.
    Call after fixing u["ext"]: flaps do not count for the size."""
    sides = [f for f in u["films"] if f["kind"] == "side"]
    if not sides:
        return 0
    f = sides[0]
    P = f["P"]
    nl, nc = P.shape[:2]
    if nl < 3:
        return 0
    cen = P.reshape(-1, 3).mean(0)
    n = 0
    for _ in range(int(rng.integers(1, 4))):
        w = int(rng.integers(max(3, nc // 40), max(4, nc // 8)))
        j0 = int(rng.integers(0, nc - w - 1))
        ln = int(rng.integers(2, min(6, nl) + 1))
        i1 = nl - 1 if rng.random() < 0.6 else int(rng.integers(ln - 1, nl))
        i0 = max(0, i1 - ln + 1)
        Q = P[i0:i1 + 1, j0:j0 + w + 1].copy()
        if Q.shape[0] < 2 or Q.shape[1] < 2:
            continue
        out = Q[..., :2] - cen[:2]
        out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-9)
        t = np.linspace(1, 0, Q.shape[0])[:, None] if rng.random() < 0.75 else np.linspace(0, 1, Q.shape[0])[:, None]
        amp = rng.uniform(0.02, 0.10)
        Q[..., :2] += out * (amp * t ** 1.5)[..., None] + rng.normal(0, 0.004, Q[..., :2].shape)
        Q[..., 2] -= 0.3 * amp * t ** 2  # sags a little
        N = U.grid_normals(Q)
        N *= np.where(((Q[..., :2] - cen[:2]) * N[..., :2]).sum(-1, keepdims=True) < 0, -1, 1)
        u["films"].append(dict(P=Q, N=N, uv=U.grid_uv(Q), kind="flap",
                               **{k: f[k] for k in ("style", "tone", "base", "gain")}))
        n += 1
    return n


def edge_label(rng, u, H):
    """v7: shipping label near a corner (60 %) or an edge of the top, also on cages (on the top rail, like a tag)
    -> symbol corners (4, 3) or None. A big label may overhang the edge a little (not part of the size)."""
    pts = np.concatenate([p["P"] for p in u["parts"] if not p.get("carrier") and p.get("mat") != "strap"]
                         + [f["P"].reshape(-1, 3) for f in u["films"] if f["kind"] == "top"])
    top = pts[pts[:, 2] > H - 0.03]
    if len(top) < 3:
        return None
    lo, hi = top[:, :2].min(0), top[:, :2].max(0)
    ctr, half = (lo + hi) / 2, (hi - lo) / 2
    s = rng.choice([-1.0, 1.0], 2)
    tgt = ctr + s * np.maximum(half - rng.uniform(0.10, 0.20, 2), 0.0)  # label centre: stays (mostly) on the top
    if rng.random() >= 0.6:  # along an edge instead of at a corner
        k = int(rng.integers(2))
        tgt[k] = ctr[k] + rng.uniform(-0.6, 0.6) * half[k]
    x, y = top[np.argmin(np.hypot(top[:, 0] - tgt[0], top[:, 1] - tgt[1])), :2]
    near = np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 0.09
    z = float(pts[near, 2].max()) + 0.001
    fz = U.film_top_z(u["films"], x, y)
    if fz is not None:
        z = max(z, fz + 0.002)
    yaw = rng.integers(4) * math.pi / 2 + rng.normal(0, 0.08) if rng.random() < 0.7 else None
    part, dmc = U.label_part(rng, float(x), float(y), z, yaw=yaw, max_size=0.16)
    u["parts"].append(part)
    return dmc


def damage_label(rng, part, dmc):
    """v7: creased / torn / scuffed / dirty label, sometimes lifted at one side (tilted). -> new symbol corners."""
    t = np.array(part["top_tex"], np.float32)
    h, w = t.shape
    tone = float(np.median(t))
    if rng.random() < 0.6:  # creases: a thin bright / dark line with a shaded side
        for _ in range(int(rng.integers(1, 4))):
            p0, p1 = rng.uniform(0, 1, 2) * (w, h), rng.uniform(0, 1, 2) * (w, h)
            m = np.zeros_like(t)
            cv2.line(m, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1.0, int(rng.integers(1, 4)))
            t += m * rng.choice([-1, 1]) * rng.uniform(15, 40)
    if rng.random() < 0.4:  # torn-off corner: the carton shows through
        cx, cy = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)][int(rng.integers(4))]
        a, b = rng.uniform(0.1, 0.35) * w, rng.uniform(0.1, 0.35) * h
        tri = np.array([[cx, cy], [cx + (a if cx == 0 else -a), cy], [cx, cy + (b if cy == 0 else -b)]], np.int32)
        cv2.fillPoly(t, [tri], rng.uniform(80, 170))
    if rng.random() < 0.5:  # scuffs / scratches over the symbol
        for _ in range(int(rng.integers(3, 15))):
            c = tuple(int(v) for v in rng.uniform(0, 1, 2) * (w, h))
            cv2.ellipse(t, c, (int(rng.integers(1, 6)), int(rng.integers(1, 4))), rng.uniform(0, 180), 0, 360,
                        tone if rng.random() < 0.7 else rng.uniform(20, 80), -1)
    if rng.random() < 0.4:  # dirt smudge
        yy, xx = np.mgrid[0:h, 0:w]
        c, r = rng.uniform(0, 1, 2) * (w, h), rng.uniform(0.1, 0.4) * max(w, h)
        t -= rng.uniform(15, 60) * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * r * r))
    part["top_tex"] = np.clip(t, 0, 255)
    if rng.random() < 0.3:  # lifted along one side
        a = rng.uniform(0, 2 * math.pi)
        R = U.rotz(a) @ U.rotx(math.radians(rng.uniform(3, 12))) @ U.rotz(-a)
        c = part["P"].mean(0)
        P2, dmc = (part["P"] - c) @ R.T + c, (dmc - c) @ R.T + c
        lift = part["P"][:, 2].min() - P2[:, 2].min()
        part["P"], dmc = P2 + [0, 0, lift], dmc + [0, 0, lift]
    return dmc


def label_v7(rng, u, H):
    """v7 label: near an edge / corner (50 %, always on cages), else as in v5; stickers damaged half the time."""
    n0 = len(u["parts"])
    if u["kind"] == "cage" or rng.random() < V7_EDGE_LABEL_P:
        dmc = edge_label(rng, u, H)
    else:
        dmc = V3.add_label(rng, u, H)
    if dmc is not None and len(u["parts"]) > n0 and rng.random() < V7_LABEL_DAMAGE_P:
        dmc = damage_label(rng, u["parts"][-1], np.asarray(dmc, np.float64))
    return dmc


def make_unit(rng, hmax, kind=None):
    """One labelled shipping unit (local frame) no taller than hmax, with size / tone variation."""
    for _ in range(25):
        r = rng.random() if kind is None else {"parcel": 0.95}.get(kind, 0.0)
        if V6 and kind is None and r < V6_CAGE_P + V6_OVERHANG_P:
            u = cage_unit(rng) if r < V6_CAGE_P else overhang_unit(rng)
            if u is not None and V3.unit_extent(u)[2] <= hmax:
                return u
            continue
        if V6 and kind is None:  # the rest keeps the v5 mix
            r = (r - V6_CAGE_P - V6_OVERHANG_P) / (1 - V6_CAGE_P - V6_OVERHANG_P)
        if r < 0.40:
            u = V3.v2_unit(rng)
        elif r < 0.72:
            u = U.make_unit(rng)
        elif r < 0.80:
            u = double_stack(rng, hmax)
        elif r < 0.88:
            u = small_pallet(rng)
        else:
            u = parcel(rng, big=kind is None)
        if u is None:
            continue
        if u["kind"] not in ROUND and u["kind"] != "cage" and rng.random() < 0.5:
            s = rng.uniform(0.85, 1.2)
            scale_unit(u, s * rng.uniform(0.92, 1.08), s * rng.uniform(0.92, 1.08), rng.uniform(0.8, 1.25))
        if rng.random() < 0.5:
            retone(rng, u)
        if V7:
            u["damage"] = damage_load(rng, u) if rng.random() < V7_DAMAGE_P else []
            if u["films"] and rng.random() < V7_FLAP_P:
                u["ext"] = V3.unit_extent(u)
                if film_flaps(rng, u):
                    u["damage"].append("film_flaps")
        if unit_ext(u)[2] <= hmax:
            return u
    return None


# ---------------------------------------------------------------------------------- placement
class State:
    def __init__(self, rng, cam, mask):
        self.rng, self.cam, self.mask = rng, cam, mask
        self.S = V3.SceneBuf(cam, rng)
        self.taken, self.objects, self.cut = [], [], 0
        self.placed = []  # labelled units: (cx, cy, yaw, footprint l, w) -- v6 puts floor mats under some
        self.hmax = float(min(2.6, cam.mount_h - 1.3))
        self.ext = E.view_extent(cam)

    def add_thing(self, th):
        self.S.add_parts(E.cap_parts(th.parts), th.yaw, th.cx, th.cy)
        if th.quads:
            self.S.add_quads(self.rng, th.quads, th.yaw, th.cx, th.cy)
        if th.films:
            self.S.add_films(th.films, th.yaw, th.cx, th.cy)
        if th.foot is not None:
            self.taken.append(th.foot)

    def inside(self, P):
        """All points (N, 3) project into the valid image."""
        uv = E.project_world(self.cam, P)
        W, H = self.cam.size
        if not ((uv >= 2).all() and (uv[:, 0] <= W - 3).all() and (uv[:, 1] <= H - 3).all()):
            return False
        return bool(self.mask[np.round(uv[:, 1]).astype(int), np.round(uv[:, 0]).astype(int)].all())

    def cut_ok(self, x, y, box):
        """Centre outside the frame but part of the unit inside it (~ a unit cut by the image border)."""
        u, v = E.project_world(self.cam, [[x, y, 0.0]])[0]
        W, H = self.cam.size
        if 0 <= u < W and 0 <= v < H:
            return False
        uv = E.project_world(self.cam, box)
        return uv[:, 0].max() > W * 0.03 and uv[:, 0].min() < W * 0.97 and uv[:, 1].max() > H * 0.03 and \
            uv[:, 1].min() < H * 0.97

    def random_point(self, margin=0.0):
        x0, y0, x1, y1 = self.ext
        # an inset (margin < 0) wider than half the view collapses to the centre line instead of failing
        ax_, bx = (x0 - margin, x1 + margin) if x1 + margin >= x0 - margin else ((x0 + x1) / 2,) * 2
        ay, by = (y0 - margin, y1 + margin) if y1 + margin >= y0 - margin else ((y0 + y1) / 2,) * 2
        return self.rng.uniform(ax_, bx), self.rng.uniform(ay, by)

    def place_unit(self, u, cx, cy, yaw, gap, truck_p, allow_cut=True):
        """Place u if its footprint (+ pallet truck) is free; labelled when fully visible, unlabelled when cut by the
        image edge with the centre outside. -> True if placed."""
        rng = self.rng
        lo, hi, H = unit_ext(u)
        foot_local = rect_corners(*(lo + hi) / 2, *(hi - lo), 0.0)
        foot = G.transform(np.c_[foot_local, np.zeros(4)], yaw, cx, cy)[:, :2]
        truck = None
        feet = [foot]
        if u["pallet"] and u["kind"] != "cage" and rng.random() < truck_p:
            L = u["pallet"][0]
            end = 1 if rng.random() < 0.5 else -1
            root = L / 2 + (rng.uniform(0.0, 0.08) if rng.random() < 0.75 else rng.uniform(0.1, 0.5))
            truck = (U.build_truck(rng), end, root, rng.normal(0, 0.02))
            tl = np.array([[end * root, -0.32], [end * (root + 1.3), -0.32], [end * (root + 1.3), 0.32],
                           [end * root, 0.32]])
            feet.append(G.transform(np.c_[tl, np.zeros(4)], yaw, cx, cy)[:, :2])
        if not all(E.free(f, self.taken, gap) for f in feet):
            return False
        box = np.concatenate([np.c_[foot, np.zeros(4)], np.c_[foot, np.full(4, H)]])
        label = self.inside(box)
        if not label:
            if not allow_cut or not self.cut_ok(cx, cy, box):
                return False
        elif len(self.objects) >= MAX_LABELLED:
            return False
        if V7:
            dmc = label_v7(rng, u, H) if rng.random() < 0.85 else None
        else:
            dmc = V3.add_label(rng, u, H) if (u["kind"] != "cage" and rng.random() < 0.85) else None
        S = self.S
        S.add_parts(u["parts"], yaw, cx, cy)
        if truck:
            parts, end, root, dy = truck
            tx, ty = G.transform(np.array([[end * root, dy, 0.0]]), yaw, cx, cy)[0, :2]
            S.add_parts(parts, yaw + (math.pi if end > 0 else 0.0), tx, ty)
        S.add_quads(rng, u["quads"], yaw, cx, cy)
        S.add_films(u["films"], yaw, cx, cy)
        self.taken += feet
        if not label:
            self.cut += 1
            return True
        self.placed.append((*G.transform(np.array([[*(lo + hi) / 2, 0.0]]), yaw, cx, cy)[0, :2], yaw, *(hi - lo)))
        cam = self.cam
        bottom, top = V3.w2c(cam, np.c_[foot, np.zeros(4)]), V3.w2c(cam, np.c_[foot, np.full(4, H)])
        pal_top = None
        if u["pallet"]:
            pal_top = V3.w2c(cam, G.transform(np.c_[rect_corners(0, 0, *u["pallet"], 0), np.full(4, G.PALLET_H)],
                                              yaw, cx, cy))
        self.objects.append({
            "name": f"unit_{len(self.objects):02d}",
            "type": u["type"],
            "object_dimensions_m": {"width_m": float(np.linalg.norm(foot[1] - foot[0])),
                                    "length_m": float(np.linalg.norm(foot[3] - foot[0])), "height_m": H},
            "boxes_3d_cam": np.round(np.concatenate([bottom, top]), 6).tolist(),
            "synth": {"kind": u["kind"], "top_face_3d": np.round(top, 6).tolist(),
                      "pallet_top_3d": np.round(pal_top, 6).tolist() if pal_top is not None else None,
                      "dmc_3d": np.round(V3.w2c(cam, G.transform(dmc, yaw, cx, cy)), 6).tolist()
                      if dmc is not None else None,
                      "label_m": G.LABEL_M, "load_lw_m": u["load"], "load_h_m": H - u["carrier_h"],
                      "pallet_lw_m": u["pallet"], "truck": truck is not None,
                      "film": u["films"][0]["style"] if u["films"] else None,
                      "soft": any(p.get("soft") for p in u["parts"]), "damage": u.get("damage", [])},
        })
        return True


def layout_rows(st, ax, n_rows=None, v_start=None, truck_p=0.15):
    """Units side by side along rows parallel to the floor axis; the rows run through and beyond the frame."""
    rng = st.rng
    R = E.rot2(ax)
    corners = np.array([[st.ext[0], st.ext[1]], [st.ext[2], st.ext[1]], [st.ext[2], st.ext[3]],
                        [st.ext[0], st.ext[3]]]) @ R  # view in layout coordinates (rows of R^T)
    u0, u1 = corners[:, 0].min() - 1.5, corners[:, 0].max() + 1.5
    vs = np.sort(corners[:, 1])  # inner band of the (rotated) view across the rows
    # narrow inner band (< ~11 cm): clamp the range instead of failing (same draw as before for every other scene)
    v = rng.uniform(vs[1] + 0.05, max(vs[1] + 0.05, vs[1] + 0.45 * (vs[2] - vs[1]))) if v_start is None else v_start
    for _ in range(int(rng.choice([1, 2, 2, 3, 3])) if n_rows is None else n_rows):
        r = rng.random()
        gaps = (0.0, 0.03) if r < 0.35 else (0.05, 0.4) if r < 0.75 else (0.4, 1.2)
        turn_p = rng.choice([0.0, 0.5, 1.0])
        cursor, depth = u0 + rng.uniform(0, 1.0), 0.0
        while cursor < u1:
            u = make_unit(rng, st.hmax)
            if u is None:
                break
            lo, hi, _ = V3.unit_extent(u)
            turn = rng.random() < turn_p
            ext = hi - lo
            eu, ev = (ext[1], ext[0]) if turn else (ext[0], ext[1])
            jit = rng.normal(0, 0.015)
            yaw = ax + (math.pi / 2 if turn else 0.0) + (math.pi if rng.random() < 0.5 else 0.0) + jit
            slack = abs(math.sin(jit)) * max(eu, ev)  # corners swing out when the unit is not quite straight
            wc = R @ np.array([cursor + slack + eu / 2, v + ev / 2 + rng.normal(0, 0.02)])
            c = wc - E.rot2(yaw) @ ((lo + hi) / 2)
            if st.place_unit(u, c[0], c[1], yaw, -0.004, truck_p):
                depth = max(depth, ev)
            cursor += eu + 2 * slack + 0.005 + rng.uniform(*gaps)
        v += max(depth, 0.8) + (rng.uniform(0.0, 0.3) if rng.random() < 0.35 else rng.uniform(0.5, 1.8))


def layout_scatter(st, n, gap, truck_p, kind=None, spread=1.0):
    rng = st.rng
    cx0, cy0 = st.random_point(-0.8)
    for _ in range(n):
        u = make_unit(rng, st.hmax, kind)
        if u is None:
            continue
        lo, hi, H = V3.unit_extent(u)
        inset = -(math.hypot(*(hi - lo)) / 2 + 0.05)
        for _try in range(60):
            if spread < 1.0 and _try < 30:
                x, y = cx0 + rng.normal(0, spread), cy0 + rng.normal(0, spread)
            else:
                x, y = st.random_point(inset)
            yaw = rng.uniform(0, 2 * math.pi) if rng.random() < 0.5 else \
                rng.choice([0, math.pi / 2]) + rng.normal(0, 0.1)
            if st.place_unit(u, x, y, yaw, gap, truck_p, allow_cut=False):
                break


def cut_units(st, n):
    """Units standing across the image border (centre outside, never labelled)."""
    rng = st.rng
    x0, y0, x1, y1 = st.ext
    for _ in range(n):
        u = make_unit(rng, st.hmax)
        if u is None:
            continue
        for _try in range(25):
            side = int(rng.integers(4))
            d = rng.uniform(0.1, 0.9)
            x, y = st.random_point(0)
            x, y = [(x0 - d, y), (x1 + d, y), (x, y0 - d), (x, y1 + d)][side]
            if st.place_unit(u, x, y, rng.uniform(0, 2 * math.pi), 0.05, 0.3):
                break


def place_free(st, parts, l, w, gap=0.1, margin=0.5, tries=30, yaw=None, xy=None):
    rng = st.rng
    for _ in range(tries):
        x, y = st.random_point(margin) if xy is None else xy
        a = rng.uniform(0, 2 * math.pi) if yaw is None else yaw
        foot = E.world_rect(x, y, l, w, a)
        if E.free(foot, st.taken, gap):
            st.add_thing(E.Thing(parts, yaw=a, cx=x, cy=y, foot=foot))
            return True
    return False


def place_extent_thing(st, parts, x0, x1, wdt, gap=0.1, tries=30):
    """Thing whose local footprint spans x0..x1 (asymmetric, e.g. forklift)."""
    rng = st.rng
    for _ in range(tries):
        x, y = st.random_point(0.8)
        a = rng.uniform(0, 2 * math.pi)
        c = E.rot2(a) @ [(x0 + x1) / 2, 0.0] + (x, y)
        foot = E.world_rect(c[0], c[1], x1 - x0, wdt, a)
        if E.free(foot, st.taken, gap):
            st.add_thing(E.Thing(parts, yaw=a, cx=x, cy=y, foot=foot))
            return True
    return False


# ---------------------------------------------------------------------------------- lighting
def sample_lights(rng, cam, doors):
    """Ambient + lamps above the camera (+ skylights, sun beams through openings, light from the dock doors).
    Rows: x, y, z, size_x, size_y, power, target x, y, z, spread (rad). Normalised as in v3: a white horizontal
    surface at the image centre gets ~1.0 from ambient + lamps (beams and door light come on top)."""
    style = rng.choice(["hall", "night", "day"], p=[0.5, 0.2, 0.3])
    amb = {"hall": rng.uniform(0.35, 0.85), "night": rng.uniform(0.06, 0.3), "day": rng.uniform(0.5, 0.9)}[style]
    lamps = []
    zc = cam.mount_h
    if style != "day" or rng.random() < 0.5:
        dx, dy = rng.uniform(2.5, 7), rng.uniform(2.5, 7)
        z = zc + rng.uniform(0.3, 3.0)
        sx, sy = (rng.uniform(0.2, 0.7), rng.uniform(0.1, 0.4)) if style == "night" else \
            (rng.uniform(0.3, 1.6), rng.uniform(0.1, 0.6))
        a, ox, oy = rng.uniform(0, math.pi), rng.uniform(0, dx), rng.uniform(0, dy)
        on = rng.uniform(0.6, 1.0)
        for i in range(-4, 5):
            for j in range(-4, 5):
                x, y = ox + i * dx, oy + j * dy
                x, y = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
                if math.hypot(x, y) < 12 and rng.random() < on:
                    lamps.append([x, y, z, sx, sy])
    if style == "day":  # skylights / roof light bands
        for _ in range(int(rng.integers(1, 5))):
            lamps.append([rng.uniform(-6, 6), rng.uniform(-6, 6), zc + rng.uniform(2, 5), rng.uniform(1, 3),
                          rng.uniform(3, 10)])
    if not lamps:
        lamps.append([0.0, 0.0, zc + 1.0, 1.0, 1.0])
    lamps = np.array(lamps)
    d2 = (lamps[:, :3] ** 2).sum(1)
    w = rng.uniform(0.5, 1.5, len(lamps))
    contrib = w * (lamps[:, 2] ** 2 / d2) / (math.pi ** 2 * d2)
    power = w * (1 - amb) / contrib.sum()
    rows = [[x, y, z, sx, sy, P, x, y, 0.0, math.pi] for (x, y, z, sx, sy), P in zip(lamps, power)]
    n_beams = int(rng.integers(1, 3)) if (style == "day" and rng.random() < 0.5) or rng.random() < 0.06 else 0
    for _ in range(n_beams):  # near-parallel sun beam through a window / skylight / door: bright, sharp patch
        el, az = rng.uniform(0.5, 1.3), rng.uniform(0, 2 * math.pi)
        d = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        tgt = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), 0.0])
        src = tgt + d * rng.uniform(10, 16)
        sx, sy = rng.uniform(0.8, 3.0), rng.uniform(0.4, 2.0)
        E_rel = rng.uniform(0.5, 3.0)  # irradiance relative to the nominal light
        rows.append([*src, sx, sy, BEAM_K * E_rel * sx * sy, *tgt, math.radians(rng.uniform(1.0, 4.0))])
    for x, y, nx, ny in doors:  # soft daylight / yard light coming in through open dock doors
        if rng.random() < 0.6:
            src = np.array([x + nx * rng.uniform(2, 5), y + ny * rng.uniform(2, 5), rng.uniform(1.5, 3.0)])
            dd = float(np.linalg.norm(src))
            share = rng.uniform(0.05, 0.4)
            rows.append([*src, rng.uniform(2.5, 3.5), rng.uniform(2.5, 3.5),
                         share * math.pi ** 2 * dd * dd / max(src[2] / dd, 0.2), x, y, 0.0, math.pi])
    return style, amb, np.array(rows, np.float64)


BEAM_K = 3.6  # area-light power per m^2 of a 1-4 deg beam giving ~1x the nominal irradiance (calibrated render)


# ---------------------------------------------------------------------------------- scene
def make_scene(seed, samples):
    rng = np.random.default_rng(seed)
    cam = E.make_camera(rng)
    set_scales(rng, cam)
    st = State(rng, cam, E.valid_mask(cam))
    names = list(MODES)
    mode = names[rng.choice(len(names), p=np.array(list(MODES.values())) / sum(MODES.values()))]
    ax = rng.uniform(0, math.pi) if rng.random() < 0.3 else rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.03)
    R = E.rot2(ax)
    layout = {"axis": ax, "levellers": [], "wall": None}
    top = cam.mount_h + 2.0
    corners = np.array([[st.ext[0], st.ext[1]], [st.ext[2], st.ext[1]], [st.ext[2], st.ext[3]],
                        [st.ext[0], st.ext[3]]]) @ R
    vmin, vmax = corners[:, 1].min(), corners[:, 1].max()
    doors_world = []

    # ---- fixed structures
    if mode == "dock" or rng.random() < 0.15:
        # wall on the far side of the frame, never over the camera (walls / columns reach above it)
        yw = rng.uniform(min(0.8, vmax), max(vmax + 0.3, 0.9))
        things, lev, doors = E.wall_with_doors(rng, cam, ax, yw, top)
        for th in things:
            st.add_thing(th)
        layout["levellers"] = lev
        layout["wall"] = (yw, rng.choice(["dark", "trailer", "yard"], p=[0.5, 0.25, 0.25]))
        for x in doors:
            p = R @ [x, yw + 0.3]
            doors_world.append((p[0], p[1], *(R @ [0.0, 1.0])))
    if rng.random() < 0.1:  # rack along one side of the frame
        s = rng.choice([-1, 1])
        v = (vmin if s < 0 else vmax) + rng.uniform(-0.8, 0.2) * s
        a, b = R @ [-12, v], R @ [12, v]
        if E.free(E.world_rect(*(R @ [0, v + s * 0.6]), 24, 1.4, ax), st.taken, 0.05):
            st.add_thing(E.rack(rng, cam, a, b, rng.uniform(1.0, 1.2), s))
    for _ in range(int(rng.integers(1, 3)) if rng.random() < 0.3 else 0):
        x, y = st.random_point(0.5)
        th = E.column(rng, x, y, top, ax)
        if E.free(th.foot, st.taken, 0.1) and E.free(th.foot, [E.world_rect(0, 0, 0.6, 0.6, 0.0)], 0.2):
            st.add_thing(th)
    if rng.random() < 0.04:
        st.add_thing(E.overhead_pipe(rng, cam))

    # ---- labelled units
    if mode == "isolated":
        layout_scatter(st, int(rng.integers(1, 3)), rng.uniform(0.6, 1.5), 0.5)
    elif mode == "rows":
        layout_rows(st, ax)
    elif mode == "cluster":
        layout_scatter(st, int(rng.integers(3, 10)), rng.uniform(0.0, 0.3), 0.15,
                       spread=rng.uniform(0.6, 1.5) if rng.random() < 0.5 else 1.0)
    elif mode == "dock":
        yw = layout["wall"][0]
        layout_rows(st, ax, n_rows=1, v_start=yw - 0.5 - rng.uniform(0.0, 2.5) - 1.3, truck_p=0.3)
        layout_scatter(st, int(rng.integers(0, 4)), rng.uniform(0.1, 0.6), 0.3)
    elif mode == "parcels":
        layout_scatter(st, int(rng.integers(5, 15)), rng.uniform(0.0, 0.15), 0.0, kind="parcel",
                       spread=rng.uniform(0.4, 1.2) if rng.random() < 0.6 else 1.0)
    cut_units(st, int(rng.integers(0, 4)) if mode != "isolated" else int(rng.integers(0, 2)))

    # ---- long fixtures where space is left, movable things (never labelled)
    if rng.random() < 0.15:
        v = rng.uniform(vmin, vmax)
        th = E.conveyor(rng, R @ [-12, v], R @ [12, v])
        if E.free(th.foot, st.taken, 0.05):
            st.add_thing(th)
    if rng.random() < 0.3:
        v = rng.uniform(vmin, vmax)
        u0 = rng.uniform(-8, 0)
        th = E.guard_rail(rng, R @ [u0, v], R @ [u0 + rng.uniform(2, 12), v])
        if E.free(th.foot, st.taken, 0.1):
            st.add_thing(th)
    if rng.random() < 0.22:
        parts, (x0, x1, wd) = E.forklift(rng)
        place_extent_thing(st, parts, x0, x1, wd)
    if rng.random() < 0.12:
        parts, (x0, x1, wd) = E.pallet_jack(rng)
        place_extent_thing(st, parts, x0, x1, wd)
    if rng.random() < 0.25:
        place_extent_thing(st, U.build_truck(rng), -0.4, 1.2, 0.65)
    for _ in range(int(rng.integers(1, 4)) if rng.random() < 0.3 else 0):
        parts, r = E.person(rng)
        place_free(st, parts, 2 * r, 2 * r, gap=0.1)
    for _ in range(int(rng.integers(1, 4)) if rng.random() < 0.3 else 0):
        parts, (l, w) = E.empty_pallets(rng)
        place_free(st, parts, l, w, gap=rng.uniform(0.0, 0.3), margin=0.8)
    for _ in range(int(rng.integers(1, 6)) if rng.random() < 0.55 else 0):
        parts, (l, w) = E.misc_item(rng)
        place_free(st, parts, l, w, gap=0.05)
    if rng.random() < 0.3:
        st.S.add_parts(V3.floor_clutter(rng), 0.0, 0.0, 0.0)
    if V6:
        layout["mats"], layout["mat_seed"] = floor_mats(st, ax), int(rng.integers(2 ** 31))

    floor, rough, finfo = E.make_floor(rng, cam, layout)
    style, amb, lights = sample_lights(rng, cam, doors_world)
    scene = st.S.arrays()
    scene.update(cam=np.array([cam.fx, cam.fy, cam.cx, cam.cy, cam.size[0], cam.size[1], cam.mount_h]),
                 lights=lights.reshape(-1, 10), world=np.float64(amb), samples=np.int64(samples),
                 seed=np.int64(seed), floor=floor.astype(np.float16), floor_rough=np.float64(rough),
                 film_params=np.array([rng.uniform(25, 80), rng.uniform(0.5, 1.0), rng.uniform(0.03, 0.12),
                                       rng.uniform(0.8, 1.0), rng.uniform(15, 50), rng.uniform(0.02, 0.06),
                                       rng.uniform(0.4, 0.6)]))
    meta = {"mode": mode, "light": style, "lens": cam.lens, "mount_h": cam.mount_h, "intrinsics_full": cam.full_intrinsics,
            "cut_units": st.cut, **finfo}
    if V6:
        meta["mats"] = len(layout["mats"])
    return scene, st.objects, meta


def floor_mats(st, ax):
    """v6: 0-2 mats / floor scales / rugs; 60 % lie under a labelled unit (as in the real frames), the rest anywhere
    in view (also in empty scenes, as negatives). -> [(cx, cy, l, w, yaw)]"""
    rng = st.rng
    if rng.random() >= V6_MAT_P:
        return []
    mats = []
    for _ in range(1 if rng.random() < 0.75 else 2):
        if st.placed and rng.random() < 0.6:
            x, y, yaw, l, w = st.placed[int(rng.integers(len(st.placed)))]
            l, w = l + rng.uniform(0.2, 1.6), w + rng.uniform(0.2, 1.4)
            x, y = x + rng.normal(0, 0.25), y + rng.normal(0, 0.25)
            yaw = yaw + rng.normal(0, 0.15) if rng.random() < 0.5 else ax + rng.normal(0, 0.03)
        else:
            x, y = st.random_point(-0.3)
            l, w = sorted([rng.uniform(0.8, 3.2), rng.uniform(0.6, 2.4)], reverse=True)
            yaw = ax + rng.choice([0.0, math.pi / 2]) + rng.normal(0, 0.03) if rng.random() < 0.7 else \
                rng.uniform(0, math.pi)
        mats.append((float(x), float(y), float(l), float(w), float(yaw)))
    return mats


# ---------------------------------------------------------------------------------- camera / compositing
def composite(rng, rgba, mask, light):
    """16-bit render (floor included, alpha ~1) -> 8-bit camera image: auto exposure, gain, gamma, noise."""
    rad = rgba[..., :3].astype(np.float32).mean(-1) / 65535 / EXPOSURE
    out = rad * ALB
    med = float(np.median(out[mask])) if mask.any() else 60.0
    out *= (rng.uniform(30, 95) / max(med, 1.0)) ** rng.uniform(0.6, 1.0)  # (partial) auto exposure
    out *= rng.uniform(0.85, 1.15)
    h, w = out.shape
    if rng.random() < 0.4:  # vignetting
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        out *= 1 - rng.uniform(0.05, 0.3) * r2 / 2
    out = 255 * np.clip(out / 255, 0, 1) ** rng.uniform(0.85, 1.18)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.8))
    shot = rng.uniform(0.004, 0.03) if light == "night" else rng.uniform(0.001, 0.015)
    out += rng.normal(0, 1, out.shape) * np.sqrt(shot * np.clip(out, 0, None) + rng.uniform(0.05, 1.0))
    out[~mask] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------- orchestration
def paths(out, i):
    shard = os.path.join(out, f"shard_{i // SHARD * SHARD:08d}")
    return os.path.join(shard, f"{i:08d}.gray.png"), os.path.join(shard, f"{i:08d}.json")


def build_job(i, seed, work, samples):
    try:
        scene, objects, meta = make_scene(seed * 1_000_003 + i, samples)
        tmp = os.path.join(work, f"{i:08d}.npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **scene)
        os.replace(tmp, os.path.join(work, f"{i:08d}.npz"))
        return objects, meta
    except Exception as e:  # never block the renderer on one bad scene
        import traceback
        print(f"scene {i} failed: {e!r}\n{traceback.format_exc()}", flush=True)
        open(os.path.join(work, f"{i:08d}.skip"), "w").close()
        return None


def finish_job(i, seed, work, out, canvas, objects, meta):
    cam = E.camera_from(meta["intrinsics_full"], meta["mount_h"], meta["lens"])
    render = os.path.join(work, f"{i:08d}.render.png")
    rgba = cv2.imread(render, cv2.IMREAD_UNCHANGED)
    img = composite(np.random.default_rng([seed, i, 7]), rgba, E.valid_mask(cam), meta["light"])
    png, js = paths(out, i)
    os.makedirs(os.path.dirname(png), exist_ok=True)
    cv2.imwrite(png, pad_square(img, canvas))
    label = {"camera_intrinsics": {"fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy, "width": canvas,
                                   "height": canvas,
                                   "source_camera_type": f"JAI GO-5000M-PGE KOWA {cam.lens} (synthetic v5)",
                                   "note": "undistorted; frame placed top-left on a black square canvas"},
             "num_objects": len(objects), "objects": objects,
             "camera_pose": {"height_m": cam.mount_h, "tilt_deg": 0.0, "roll_deg": 0.0},
             "valid_region": [0, 0, cam.size[0], cam.size[1]], "seed": int(seed * 1_000_003 + i),
             "scene": {k: v for k, v in meta.items() if k != "intrinsics_full"},
             "generator": "synth/generate_v5.py" + (" --v7" if V7 else " --v6" if V6 else "")}
    with open(js + ".tmp", "w") as f:
        json.dump(label, f, indent=1)
    os.replace(js + ".tmp", js)
    os.remove(render)
    return i


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "datasets", "SYNTH_V5_DIVERSE"))
    ap.add_argument("--n", type=int, default=3000, help="number of images")
    ap.add_argument("--start", type=int, default=0, help="first image index")
    ap.add_argument("--canvas", type=int, default=1280, help="square output side; frame goes top-left")
    ap.add_argument("--workers", type=int, default=6, help="CPU processes building scenes / compositing")
    ap.add_argument("--blender", default=shutil.which("blender") or "blender")
    ap.add_argument("--blender-procs", type=int, default=2, help="concurrent Blender renderers (overlap CPU/GPU)")
    ap.add_argument("--device", default="OPTIX", help="Cycles device: OPTIX | CUDA | CPU")
    ap.add_argument("--samples", type=int, default=48, help="Cycles samples per pixel (adaptive, denoised)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-work", action="store_true", help="keep the work folder (scenes / raw renders)")
    ap.add_argument("--v6", action="store_true",
                    help="v6 scenes: floor mats / scales / rugs (unlabelled), +10 %% cages / roll containers, "
                         "+10 %% loads overhanging the pallet; default out datasets/SYNTH_V6")
    ap.add_argument("--v7", action="store_true",
                    help="v7 scenes (+ all of v6): damaged loads (cartons missing / slid / leaning), torn film flaps, "
                         "labels near edges / corners / on cages, creased / torn / scuffed / lifted labels; "
                         "default out datasets/SYNTH_V7")
    args = ap.parse_args()
    global V6, V7
    V7 = args.v7
    V6 = args.v6 or V7
    if args.out == ap.get_default("out") and (V6 or V7):
        args.out = os.path.join(ROOT, "datasets", "SYNTH_V7" if V7 else "SYNTH_V6")

    todo = [i for i in range(args.start, args.start + args.n) if not all(map(os.path.exists, paths(args.out, i)))]
    print(f"{args.n - len(todo)} done already, {len(todo)} to render", flush=True)
    if not todo:
        return
    work = os.path.join(args.out, "_work")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    with open(os.path.join(work, "todo.txt"), "w") as f:
        f.write("\n".join(map(str, todo)) + "\n")
    procs = []
    for k in range(args.blender_procs):
        log = open(os.path.join(work, f"blender_{k}.log"), "w")
        procs.append(subprocess.Popen(
            [args.blender, "-b", "--factory-startup", "-noaudio", "--python-exit-code", "1", "-P",
             os.path.join(HERE, "blender_render_v5.py"), "--", "--work", work, "--worker", str(k),
             "--nworkers", str(args.blender_procs), "--device", args.device],
            stdout=log, stderr=subprocess.STDOUT))

    from multiprocessing import Pool
    pool = Pool(args.workers)
    window = max(8, 2 * args.workers)
    built, finishing = {}, []
    nxt, done, t0 = 0, 0, time.time()
    try:
        while done < len(todo):
            while nxt < len(todo) and nxt - done < window:
                built[todo[nxt]] = pool.apply_async(build_job, (todo[nxt], args.seed, work, args.samples))
                nxt += 1
            progressed = False
            for i in list(built):
                r = built.get(i)
                if r is None or not r.ready():
                    continue
                res = r.get()
                if res is None:
                    del built[i]
                    finishing.append((i, None))
                    progressed = True
                elif os.path.exists(os.path.join(work, f"{i:08d}.render.png")):
                    del built[i]
                    finishing.append((i, pool.apply_async(finish_job, (i, args.seed, work, args.out, args.canvas,
                                                                       *res))))
                    progressed = True
            while finishing and (finishing[0][1] is None or finishing[0][1].ready()):
                i, r = finishing.pop(0)
                if r is not None:
                    r.get()
                done += 1
                if done % 25 == 0 or done == len(todo):
                    dt = time.time() - t0
                    print(f"{done}/{len(todo)}  {dt / done:.2f} s/img  eta {dt / done * (len(todo) - done) / 60:.0f} min",
                          flush=True)
            if not progressed and done < len(todo):  # the last image may have just been finished above
                codes = [p.poll() for p in procs]
                # a renderer may write its last image and exit between the scan above and poll(): rescan before failing
                late = any(os.path.exists(os.path.join(work, f"{i:08d}.render.png")) for i in built)
                if any(c not in (None, 0) for c in codes) or (None not in codes and not finishing and not late):
                    raise RuntimeError(f"Blender exited (codes {codes}), see {work}/blender_*.log")
                time.sleep(0.05)
    finally:
        open(os.path.join(work, "STOP"), "w").close()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        pool.terminate()
    if not args.keep_work:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
