# package_dims — 3D size of packages from the overhead camera cam_000CDF0AD90F

Single overhead monochrome camera (JAI GO-5000M, 2560x2048, mounted 5.70 m, looking straight
down), no depth sensor. Goal: length x width x height of the shipping unit (pallet included),
CPU inference, offline.

Plan: train the existing single-image 3D detector
`~/code_ws/study/3D_detection/custom_3d_model` (oriented box + metric W/L/H from one grayscale
image + intrinsics) on **synthetic images of this exact camera**, and test on the real frames.

## Layout

```
camera.py                 camera model: intrinsics, distortion, project/backproject, square padding
synth/generate.py         procedural renderer -> synthetic dataset (CPU, ~0.1 s/image)
synth/generate_v2.py      v2: z-buffered procedural renderer, part-based scenes (CPU)
synth/generate_v3.py      path-traced generator (Blender Cycles, GPU, via synth/blender_render.py) -> dataset v4
synth/units_v3.py         shipping units beyond cartons (drums, sacks, tyres, film-wrapped ...), soft cartons, film
synth/generate_v5.py      v5: diverse cameras / floors / surroundings / layouts / units (Blender, via blender_render_v5.py)
synth/env_v5.py           v5 cameras (LM8/12/16 calibrations), procedural floors, warehouse structures, people, vehicles
prepare_real.py           real frames -> test set in the same framing + Readme ground truth
tools/prepare_real_nocalib.py   same test set WITHOUT the camera calibration (raw image, datasheet focal length)
tools/visualize_labels.py draw the 3D labels on images (sanity check)
tools/eval_real.py        trained checkpoint on the real frames -> box + L x W x H in cm (network + measured with the camera geometry) vs the Readme truth
tools/overlay_calib_vs_nocalib.py   side-by-side overlays: with vs without calibration (model box, DMC, sizes)
tools/run_production.py   production volume measurement (com.dachser.ilo.volume_measurement) on the real frames
tools/overlay_model_vs_production.py   side-by-side overlays: our model vs production, sizes on the boxes
tools/overlay_calib_nocalib_production.py   three panels: our model with / without calibration, production
tools/overlay_combined.py  the same three results drawn on one image (calibrated frame), sizes + volume accuracy
train.sh                  launches custom_3d_model/train.py on the synthetic data (not run yet)
assets/background_median.png   empty floor = per-pixel median of the 19 real frames
datasets/
  SYNTH_CAM_000CDF0AD90F/      3000 synthetic images + 3D ground truth   (training)
  SYNTH_CAM_000CDF0AD90F_v3/   505 path-traced images, v2 scenes (run stopped early)   (training)
  SYNTH_CAM_000CDF0AD90F_v4/   3000 path-traced images, all unit kinds (v4)           (training)
  SYNTH_V5_DIVERSE/            3500 path-traced images, varied camera / floor / scene (v5), nothing from cam_000CDF0AD90F
  SYNTH_V6/                    3000 v6 images: v5 + floor mats / scales, more cages and overhanging loads, floor fix
  REAL_CAM_000CDF0AD90F/       19 real frames + Readme sizes             (test only)
  REAL_CAM_000CDF0AD90F_nocalib/  the same 19 frames, not undistorted, datasheet intrinsics   (test only)
```

## Data format (= custom_3d_model `objects` schema, same as TASLER_MULTI / TASLER_PACK)

```
shard_XXXXXXXX/{idx:08d}.gray.png   1280x1280: undistorted 1280x1024 frame top-left, black padding below
shard_XXXXXXXX/{idx:08d}.json
  camera_intrinsics  fx, fy, cx, cy, width, height    (pinhole, for the 1280x1280 image)
  objects[]          one per shipping unit
    type                 "palletised_load" | "parcel"
    object_dimensions_m  width_m = |c1-c0|, length_m = |c3-c0|, height_m (from the floor)
    boxes_3d_cam         8 corners, OpenCV camera frame (x right, y down, z forward), metres,
                         bottom 0-3 on the floor, top 4-7; the box encloses pallet + load
    synth                extra, ignored by the loader: kind, top_face_3d, pallet_top_3d,
                         dmc_3d (DataMatrix corners), load/pallet sizes
  camera_pose        height_m 5.702, tilt 0, roll 0
  valid_region       [0, 0, 1280, 1024]
```

Real test frames have `objects: []` and a `ground_truth` block instead:
`{package, L_m, W_m, H_m, dmc_corners_px, source}` (`dmc_corners_px` already in the 1280x1280
undistorted image). Frames of package 8 have no DMC.

**Images are undistorted.** custom_3d_model ignores lens distortion, so the synthetic data is
pinhole and every real frame must go through the same `prepare_real.py` framing before
inference (undistort -> x0.5 -> pad to 1280x1280). Intrinsics of that image:

```
fx 1277.03  fy 1277.39  cx 632.45  cy 504.42
```

## What the synthetic data contains

Rendered with the real intrinsics + mounting height over the real empty floor:

| | share | notes |
|---|---|---|
| pallet loads | ~65 % | EUR 1.2x0.8 (60 %), 1.2x1.0, random; load ≈ pallet / overhanging panel / small carton on pallet; carton grid, cardboard or white panel textures, 50 % stretch film |
| loose cartons | ~20 % | 0.3-1.4 m, no pallet |
| cages (gitterbox) | ~15 % | open top, mesh walls |
| units per image | 0 (8 %), 1, 2 (~10 %) | 0 = negatives (floor / pallet truck only) |
| height | 0.15-2.25 m | real Readme range 0.73-2.17 m |
| extras | | DMC label (16x16, 94.8 mm) on 90 % of tops, pallet trucks, floor shadows, sun patch, gain/gamma/noise/blur |

Exact labels: the unit box is the OBB of pallet + load (matches the Readme convention, e.g.
package 6: L 1.6 = load overhang, W 0.8 = pallet).

## Synthetic v2 (`synth/generate_v2.py`)

Same schema/framing/label convention as v1, more realistic scenes:

- z-buffered renderer (per-pixel plane depth): forks hidden under pallet decks, gaps between
  cartons, see-through cage mesh all occlude correctly
- one light direction per scene: Lambert shading of every face, side faces darker towards the
  floor, directional + contact floor shadows
- geometry built from parts: block pallets (deck boards, cross boards, 9 blocks), loads made of
  individual cartons (uniform / ragged-top stacks, mixed columns with cartons askew on top, single
  big box, plywood crate, panel, strapped machine, small item), gitterbox cages with posts and rim
- pallet truck engaged under ~70 % of pallets (forks inside, heel, tilted drawbar + loop handle),
  as in nearly all real frames; truck is not part of the label box
- carton SKUs (brand text / oval logos sized in metres, small print, flap seam + tape, worn boxes),
  stretch film as an alpha overlay with edge-biased crinkle highlights, straps
- sun patch from the bottom-right edge in 80 % of images (real: 15/19 frames)

Tone on valid pixels (mean / sigma): real 65.9 / 36.8, v1 66.8 / 43.5, v2 65.9 / 37.1.
~1.5 CPU-s per image.

```bash
nice -n 19 python3 synth/generate_v2.py --n 3000 --workers 8 --seed 0   # -> datasets/SYNTH_CAM_000CDF0AD90F_v2
```

## Synthetic v3 (`synth/generate_v3.py` + `synth/blender_render.py`) — path-traced

Same scenes, labels, schema and framing as v2 (scene builders imported from `generate_v2.py`), but the
image is rendered with **Blender Cycles on the GPU** instead of the OpenCV quad rasteriser:

- physically based light: mostly ambient + a grid of ceiling lamps (sometimes a few lamps, a soft dock-door
  light from the bottom right), giving soft real shadows, contact occlusion under / inside pallets and
  between cartons, light bounced from the floor
- the floor is a shadow catcher textured with the real empty floor, so shadows land on the real background
  and objects are composited over it with antialiased edges
- materials: cardboard with bulging faces, darker handled rims and scuffs, rounded edges (bevel) on
  everything; painted truck; galvanised cage mesh; stretch film = clear plastic rounded over the load edges
  with patchy crinkles that catch lamp glints + v2's edge-biased milky creases
- pallet truck rebuilt from round parts (tapered forks, pump, wheels, drawbar, rounded-triangle handle)
- camera: linear, gain/gamma/sun patch/vignetting as v2, sensor noise at the real level (sigma ~0.8)

Tone on valid pixels (mean / sigma): real 65.9 / 36.2, v3 64.0 / 35.0. Blender projection checked
against `Camera.project` (< 0.1 px). ~2 s per image (RTX 4070 laptop, 2 Blender processes).
Needs Blender >= 4.2 (tested 5.2.2); the first run compiles the OptiX kernels (~3 min, cached).

```bash
# v3 was made with the plain v2 scenes; generate_v3.py now builds the v4 scenes (see below) and writes to _v4
# resumes after an interruption (skips finished images); --samples / --device CUDA|CPU / --blender PATH
```

## Synthetic v4 (`synth/generate_v3.py` + `synth/units_v3.py`) — all kinds of shipping units

Same renderer, schema, framing and label convention as v3 (box = OBB of everything incl. pallet / carrier / film),
richer scenes. A cross-dock does not only see cartons:

| share of units | what |
|---|---|
| ~60 % | v2 cartons / pallet loads / cages, upgraded: stretch film draped over the real load outline (clear 70 %, bubble wrap, black / white opaque, tinted), soft cartons in 35 % (rounded edges and corners, bulging sides, sagging / domed tops, dents), edge protectors + top sheet + straps |
| ~32 % | sacks (paper / PE, interlocked layers), big bags with loops, beverage pallets, crate stacks on dollies, roll containers, machines / furniture under film, octabins, pallet collars, polybags (loose or heaped), slatted crates, IBCs, buckets / canisters, long goods on bearers |
| ~8 % (rare) | round goods: steel / plastic drums, cable drums (standing / lying), coils, tyres and wheels on rims, paper reels, carpet rolls, mailing tubes |

- smooth curved surfaces (per-corner normals), identical items share one texture
- shipping label = sticker with the DataMatrix on the highest flat spot, on top of the film (exact `dmc_3d`);
  ~73 % of units carry one (cages, crate dollies, roll containers, tubes, slatted lids: none)
- `synth` json extras: `kind` (e.g. `stack`, `sacks`, `tyres`), `film` (style or null), `soft`
- tone on valid pixels (mean / sigma): real 65.9 / 36.8, v4 previews 67 / 37; ~2 s per image (RTX 4070 laptop)

```bash
nice -n 10 python3 synth/generate_v3.py --n 3000 --workers 6 --seed 0   # -> datasets/SYNTH_CAM_000CDF0AD90F_v4
```

## Synthetic v5 (`synth/generate_v5.py` + `synth/env_v5.py` + `synth/blender_render_v5.py`) — diverse scenes

v1-v4 are all "this camera over this floor": one real median floor image, one camera, 0-2 isolated units, so the
images look alike. v5 changes everything per image and takes **nothing from cam_000CDF0AD90F**, so its 19 frames stay
an independent test (no real floor image, not its camera model 51185017, no tone / noise / sun-patch matching to its
frames, DMC label size random 75-115 mm instead of the 94.8 mm fitted on them, no "truck under 70 % of pallets").

| | v5 |
|---|---|
| camera | JAI GO-5000M + KOWA LM8HC (20 %, 2.6-4.0 m) / LM12HC (40 %, 3.9-6.3 m, model 51185020) / LM16HC (40 %, 5.3-8.0 m); median factory calibration of the shipped model + unit jitter; framed like `prepare_real.py` incl. the black undistortion corners; intrinsics per image in the json |
| floor | procedural and rendered (lit, glossy, shadowed): concrete, worn, polished, epoxy, dark painted, asphalt, tiles; slab joints, cracks, repairs, stains, wet patches, tyre marks, wear paths; lane lines, pallet bays with numbers, hatched zones, arrows, zone codes (worn); dock levellers; dark bay / trailer / yard beyond a dock wall |
| surroundings (never labelled) | walls with dock doors (open / half / closed, seals, bollards), columns with guards, guard rails, pallet racks with loads on the beams, roller / belt conveyors, forklifts, electric pallet trucks, hand pallet trucks, people (hi-vis, helmets, walking, carrying), empty pallets and stacks, bins, cones, bollards, cables, sheets, film rolls, trolleys, desks, overhead pipes |
| layouts | isolated 17 % / rows 30 % (touching, tight or spaced, along the floor axis) / clusters 22 % / dock 12 % / loose parcels 9 % / empty 10 %; units cut by the image border (centre outside, unlabelled); up to 10 labelled units per image (mean ~3.7) |
| units | v4 units + double-stacked pallet loads, half / quarter pallets, loose parcels; 50 % size-scaled (footprint 0.85-1.2, height 0.8-1.25), 50 % re-toned (dark / bright goods); height <= min(2.6 m, camera height - 1.3 m) |
| light | hall (lamp grid above the camera), night (low ambient, lamp pools), day (skylights); sun beams through openings; daylight through open dock doors |
| image | auto exposure (frame median 30-95), gain, gamma, vignetting, blur, shot + read noise (stronger at night) |

`json["scene"]` keeps mode, lens, height, floor kind, light style and the number of cut units of every image.
~3.7 s per image (RTX 4070 laptop, 2 Blender processes, 6 CPU workers).

```bash
nice -n 10 python3 synth/generate_v5.py --n 3000 --workers 6 --seed 0      # -> datasets/SYNTH_V5_DIVERSE (resumes)
DATA="$PWD/datasets/SYNTH_V5_DIVERSE $PWD/datasets/SYNTH_CAM_000CDF0AD90F_v4" RUN=cam5_v5 ./train.sh  # v5 + v4
DATA="$PWD/datasets/SYNTH_V5_DIVERSE" RUN=v5only ./train.sh   # v5 only: fair test on the cam5 frames
```

## Synthetic v6 (`synth/generate_v5.py --v6`) — mats, cages, overhangs + floor fix

v5 scenes plus the failure cases seen on the real frames; still nothing taken from cam_000CDF0AD90F.
Without `--v6` the generator still produces exactly v5 (same scene for the same seed).

| | v6 adds |
|---|---|
| floor mats (unlabelled) | 50 % of images: 1-2 rubber mats (bumps / ribs), floor-scale platforms (checker plate), rugs, steel cover plates; dark (tone 6-80) with border bands, frames / pit edges; 60 % lie under a labelled unit (as on the real scale), the rest anywhere (also in empty scenes = negatives) |
| cages | +10 % of units: gitterbox cages resized 0.9-1.45x (up to ~1.7 x 1.2 m) or roll containers |
| overhanging loads | +10 % of units: panels / boards / big boxes longer than their pallet |
| `json` | `scene.mats` = number of mats, `generator` = `synth/generate_v5.py --v6` |

**Floor rendering fix (all renders from now on, v5 and v6).** `blender_render_v5.py` handed the 1-channel floor albedo
to Blender as `(a, 1, 1)` (only the red channel was set), so 2/3 of the floor brightness did not depend on the painted
floor: every floor feature (markings, joints, stains, mats, dark floors) was rendered with ~3x too little contrast
(measured: painted mat/floor ratio 5-7, rendered 1.1-1.4; after the fix 3.3-5.8). The albedo now goes into R, G and B.
The existing v5 images (0-3490) were rendered **before** the fix, the 9 previously missing ones and all of v6 after.

Generator fixes: two random ranges could become empty for narrow views / big units (`layout_rows`, `random_point`) and
made 9 v5 scenes fail; they are clamped now (no change for any other scene). The run loop no longer reports
"Blender exited" when the last image was finished in the same pass.

```bash
nice -n 10 python3 synth/generate_v5.py --v6 --n 3000 --workers 4 --seed 1   # -> datasets/SYNTH_V6 (resumes)
DATA="$PWD/datasets/SYNTH_V5_DIVERSE $PWD/datasets/SYNTH_V6" RUN=v56 \
    INIT="$PWD/checkpoints/v5only_3500/v5only_3500-16.ckpt" ./train.sh --epochs 20 --lr 2e-4 --num_workers 4
# -> checkpoints/v56/v56-17.ckpt (~1 h): real frames MAE L 5.3  W 8.8  H 29.0 cm, mat false detections 13 -> 1 frames
#    measured (box + camera geometry + DMC height, tools/eval_real.py): L 4.6  W 8.0  H 5.7 cm
```

## Synthetic v7 (`synth/generate_v5.py --v7`) — damaged loads, torn film, labels near edges

v6 scenes (`--v7` implies `--v6`) plus the kinds of packages that arrive in poor shape. Nothing taken from
cam_000CDF0AD90F. Without `--v7` the generator still produces exactly v5 / v6.

| | v7 adds |
|---|---|
| damaged carton loads | ~80 % chance per carton-only load (not cages, not round kinds), at least one of: 1-2 cartons missing from the top layer, an upper layer slid 2-9 cm and twisted, the whole load leaning 0.8-3°; straps dropped, stretch film re-wrapped over the new shape |
| torn film flaps | 50 % of wrapped units: 1-3 strips of side film peeling off by 2-10 cm; **not** part of the true size |
| labels near corners / edges | 50 % of labels (and all labels on cages): 10-20 cm from a corner or along an edge, on the highest point there, axis-aligned (70 %) or any yaw |
| damaged labels | 50 % of stickers: creases, torn-off corner, scuffs, dirt; 30 % of those lifted on one side (3-12° tilt; the DMC corners follow) |
| `json` | `objects[].synth.damage` = list of the damage ops, `generator` = `synth/generate_v5.py --v7` |

SYNTH_V7 (3000 images, seed 2): 1472 of 8517 units damaged (film flaps 715, layer shifted 482, leaning 439,
cartons missing 357).

```bash
nice -n 10 python3 synth/generate_v5.py --v7 --n 3000 --workers 4 --seed 2   # -> datasets/SYNTH_V7 (~1.7 h)
DATA="$PWD/datasets/SYNTH_V5_DIVERSE $PWD/datasets/SYNTH_V6 $PWD/datasets/SYNTH_V7" RUN=v567 \
    INIT="$PWD/checkpoints/v5only_3500/v5only_3500-16.ckpt" ./train.sh --epochs 20 --lr 2e-4 --num_workers 4 --w_bot 0
# -> checkpoints/v567/v567-13.ckpt (lowest val loss; ~1.5 h): real frames network MAE L 4.6  W 8.1  H 30.2 cm
#    measured --tta --snap 15: L 4.4  W 4.7  H 5.5 cm (v56: 4.4 / 5.1 / 5.7)
```

`--w_bot 0` (new `train.py` option) switches off the bottom-offset loss, so v567 is trained exactly like v56 and only
the data differ.

## How to generate a dataset

All generators write the same schema (`shard_XXXXXXXX/{idx:08d}.gray.png` + `.json`, 250 images per shard) and are
run from this folder (`~/Documents/dachser/package_dims`). For new data use **v5**.

**Requirements**
- Python 3 with `numpy`, `opencv-python` (tested 2.4 / 5.0)
- v3-v5 (path-traced): Blender >= 4.2 on `PATH` (tested 5.2.2, `/snap/bin/blender`) or `--blender /path/to/blender`,
  and an NVIDIA GPU (OptiX). The first run compiles the OptiX kernels (~3 min, cached). Without a GPU: `--device CPU`
  (much slower).
- v5 also reads `../cam_models_calib/2025.10.23_Dachser_Shipped.txt` (lens calibrations); v1-v4 read
  `../cam_000CDF0AD90F/` and `assets/background_median.png`.
- Resources for v5 with the defaults: ~3 GB GPU memory, ~6-8 GB RAM, ~0.5 MB disk per image (3000 images ~1.5 GB),
  ~4 s per image on an RTX 4070 laptop (3000 images ~3.5 h).

**1. Preview a few images first** (seconds per image, into a throwaway folder) and look at them with the labels drawn:

```bash
python3 synth/generate_v5.py --n 24 --seed 5 --out /tmp/v5_preview
python3 tools/visualize_labels.py /tmp/v5_preview /tmp/v5_preview.jpg --n 24
```

`visualize_labels.py` draws every labelled unit: green = bottom face, red = top face, blue = verticals, yellow = size.
The box must sit on the unit; units cut by the image border have no box (by design).

**2. Generate the dataset** (runs for hours: start it in the background, low CPU priority, keep a log):

```bash
nohup nice -n 10 python3 synth/generate_v5.py --n 3000 --workers 6 --seed 0 \
    --out datasets/SYNTH_V5_DIVERSE > datasets/generate_synth_v5.log 2>&1 &
tail -f datasets/generate_synth_v5.log          # "25/3000  4.06 s/img  eta 201 min"
```

**3. Stop / resume / extend.** Stop it any time (Ctrl-C / `kill`). Run the *same* command again to continue: images
whose `.png` and `.json` already exist are skipped. Every image is determined by `(--seed, index)`, so a rerun
re-creates exactly the missing ones. More images later: same `--seed` and `--out` with `--start 3000 --n 1000`; an
independent set: another `--seed` into another `--out`. The temporary `<out>/_work/` folder (scene files, raw renders,
`blender_*.log`) is deleted at the end; `--keep-work` keeps it for debugging. If the log says
`Blender exited (codes ...)`, the reason is in `<out>/_work/blender_*.log`.

**4. Check the result** (counts, then a few random shards):

```bash
ls datasets/SYNTH_V5_DIVERSE/shard_*/*.png | wc -l      # -> 3000
python3 tools/visualize_labels.py datasets/SYNTH_V5_DIVERSE /tmp/v5_check.jpg --n 16 --start 1500
```

**5. Train on it** — see the v5 section above (`DATA="$PWD/datasets/SYNTH_V5_DIVERSE" RUN=v5only ./train.sh`).

`generate_v5.py` options:

| option | default | |
|---|---|---|
| `--n`, `--start` | 3000, 0 | number of images, index of the first one |
| `--seed` | 0 | dataset seed (image i uses scene seed `seed * 1000003 + i`) |
| `--out` | `datasets/SYNTH_V5_DIVERSE` | output folder |
| `--workers` | 6 | CPU processes building scenes (~4 CPU-s each) and finishing images |
| `--blender-procs` | 2 | Blender processes rendering at the same time (overlap CPU and GPU) |
| `--samples` | 48 | Cycles samples per pixel (adaptive + denoised); more = cleaner and slower |
| `--device` | OPTIX | OPTIX / CUDA / CPU |
| `--blender` | `blender` on PATH | Blender executable |
| `--canvas` | 1280 | side of the square output image (frame top-left, black padding below) |
| `--keep-work` | off | keep `<out>/_work` |

What a scene contains (shares of layouts, lenses, floors, lights ...) is set by the constants at the top of
`synth/generate_v5.py` (`MODES`, `MAX_LABELLED`) and `synth/env_v5.py` (`LENSES`, `FLOOR_KINDS`), and by the
probabilities in `make_scene()`.

Older versions (same output schema, all tied to cam_000CDF0AD90F and its real floor):

```bash
nice -n 19 python3 synth/generate.py    --n 3000 --workers 4 --seed 0   # v1: CPU rasteriser  -> SYNTH_CAM_000CDF0AD90F
nice -n 19 python3 synth/generate_v2.py --n 3000 --workers 8 --seed 0   # v2: CPU, z-buffered -> ..._v2
nice -n 10 python3 synth/generate_v3.py --n 3000 --workers 6 --seed 0   # v4: Blender, GPU    -> ..._v4
```

## Results on the real frames (cam_000CDF0AD90F, 19 frames, 8 packages)

All errors in **cm** (predicted − true). The real frames are used **only for testing**, after training.

### Models

| model | trained on | real floor in training? | detected | L | W | H |
|:---|:---|:---:|:---:|---:|---:|---:|
| rtmdet_pack_v3 (start point) | TASLER / packstuecke | no | – | 55.1 | 30.2 | 84.3 |
| cam5_v4-08 | v4 3000 + v3 505, 10 ep | **yes** | 19/19 | 8.6 | 12.8 | 23.8 |
| cam5_v4_e20-08 | + 10 ep, lr 2e-4 | **yes** | 19/19 | 8.3 | 10.0 | **21.5** |
| v5only-18 | v5 1914, 20 ep | no | 19/19 | 8.7 | 7.3 | 36.6 |
| v5only_3500-16 | v5 3191, + 20 ep, lr 2e-4 | no | 19/19 | 7.9 | **6.9** | 33.3 |
| v56-17 | v5 3500 + v6 3000, + 20 ep, lr 2e-4 | no | 19/19 | 5.3 | 8.8 | 29.0 |
| **v567-13** | v5 3500 + v6 3000 + v7 3000, + 20 ep, lr 2e-4 | no | 19/19 | **4.6** | 8.1 | 30.2 |

Mean absolute error. v5 contains nothing from this camera (no real floor, not its calibration), so `v5only*` is the
fair test; `cam5_v4*` saw the real empty floor (`assets/background_median.png`) during training.

- v56 (v5 + v6: floor mats, more cages / overhanging loads, fixed floor rendering) is the current model:
  - false detections (orange) on the dark floor mat: **13/19 frames → 1/19** (v5only_3500 vs v56);
  - L 7.9 → **5.3** cm, H 33.3 → 29.0 cm;
  - W 6.9 → 8.8 cm: W comes out wider (mean +4.6 cm), like production (+5 to +10 cm) — mostly on stretch-wrapped
    loads whose film bulges past the 80 cm pallet;
  - package 7 (cage on the scale) still short: L 128 for 159 (box on the cage body, not the mat any more).
- Height still regresses to the mean (~150 cm): low units too high (pkg 4: 78 → ~100), tall ones too low (pkg 3: 217 → 133-184).

### Measured: model box + camera geometry (`tools/eval_real.py`)

The network's L / W / H are learnt from appearance only: they use neither the intrinsics (only the depth does) nor the
camera pose, so the network has to guess the absolute scale. The camera pose is known, though: it looks straight down
from 5.70 m. The *measured* sizes use it:

- **H** from the DMC label: `H = mount − fx · 0.0948 m / side_px` (the model's H when a frame has no label: pkg 8);
- **L × W**: the 4 corners of the model's 2D box back-projected through the intrinsics (fx, fy, cx, cy) onto the
  plane of the package top, `Z = mount − H`.

| v56-17 | L | W | H | L median | W median | H median |
|:---|---:|---:|---:|---:|---:|---:|
| network sizes | 5.3 | 8.8 | 29.0 | 4.1 | 7.4 | 27.3 |
| **measured** (camera geometry) | **4.6** | **8.0** | **5.7** | **2.8** | **7.2** | **3.5** |
| measured, without calibration | 5.2 | 7.9 | 9.0 | 3.0 | 7.5 | 5.5 |

- Package 7 (cage): 128 × 91 × 162 → **145 × 104 × 133** (true 159 × 115 × 133).
- H: 3.5 cm median; the 5.7 mean includes frame 18 (pkg 8, no label → model H, +55).
- W is now the weak side: +4.9 cm on average, also with the true H (the 2D box itself is too wide: wrap film bulging
  past the pallet, visible side faces). The training target already *is* the top face (`dataset._rbox_from_top_face`);
  on real frames the box spills over glossy wrapped side faces and onto the floor (frame 2: ~275 px wide for a
  ~231 px top face) — a sim-to-real gap of the box edges, not of the geometry.

**Tried: height from perspective (`v56bot`, not adopted).** custom_3d_model got a head for the pixel offset from the
top- to the bottom-face centre (`dbx, dby`; `model.height_from_bottom`: the bottom face is the top face shrunk towards
the principal point by `(mount − H) / mount`, so H follows from pixels + principal point + mount, without fx or a size
prior). The label target is exact (checked on the synthetic labels, also after the rot90 / zoom augmentations).
Trained 20 epochs from v56-17 (`checkpoints/v56bot/v56bot-16.ckpt`):

| perspective H | offset error | H MAE | H bias |
|:---|---:|---:|---:|
| synthetic val, units ≥ 150 px from the centre | 12 px of 36 | 23 cm | −8 cm |
| real frames, ≥ 150 px from the centre (15) | – | 64 cm | low on almost all |

The offset head learns slowly and under-estimates; on the real frames it is far worse than the DMC (3 cm) and not
better than the network's own H. v56-17 stays the model; `eval_real.py` reports `H_bot_cm` for checkpoints that have
the head and uses it only for packages without a DMC label.

### Height: DMC label vs models vs production

`H = mount − fx · 0.0948 m / side_px` from the DataMatrix label (17 frames have one).
*fair* = the label size is re-fitted leaving the tested package out (94.8 mm was fitted on these frames).
*production* = `com.dachser.ilo.volume_measurement` `calc_height` (datasheet focal length, off-centre + polynomial
correction, − 3.5 cm), re-implemented and run on the same code-scanner data.

| height method | frames | MAE | median | max | bias |
|:---|:---:|---:|---:|---:|---:|
| **DMC + calibration (ours)** | 17 | **3.1** | 3.5 | 7.4 | +2.1 |
| DMC + calibration, fair | 17 | **3.0** | 2.3 | 7.6 | −0.3 |
| DMC without calibration | 17 | 6.0 | 5.5 | 10.9 | +0.6 |
| DMC production (`calc_height`) | 17 | 9.0 | 8.3 | 22.3 | +9.0 |
| model cam5_v4_e20-08 | 19 | 21.5 | 22.8 | 49.4 | −2.3 |
| model v5only_3500-16 | 19 | 33.3 | 31.7 | 68.7 | +3.7 |
| model v56-17 | 19 | 29.0 | 27.3 | 83.6 | −7.4 |

<details>
<summary>Height error per frame (cm)</summary>

| frame | pkg | true H | DMC (ours) | DMC no calib | DMC production | model v5only_3500 | model v56 | model cam5_v4_e20 |
|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 120 | +1 | +10 | +14 | +43 | +15 | +25 |
| 1 | 1 | 120 | −4 | −11 | +8 | +38 | +25 | +2 |
| 2 | 1 | 120 | −1 | −10 | +8 | +26 | +27 | −14 |
| 3 | 1 | 120 | +5 | −5 | +20 | +12 | −4 | −17 |
| 4 | 2 | 211 | +3 | +8 | +4 | −32 | −35 | −13 |
| 5 | 2 | 211 | +7 | −1 | +9 | +11 | −8 | −2 |
| 6 | 2 | 211 | +4 | +1 | +2 | −46 | −37 | −24 |
| 7 | 2 | 211 | +5 | +2 | +4 | −45 | −49 | −29 |
| 8 | 3 | 217 | +4 | +9 | +4 | −25 | −33 | −19 |
| 9 | 3 | 217 | +4 | +3 | +4 | −69 | −84 | −9 |
| 10 | 4 | 78 | −4 | −8 | +15 | +50 | +19 | +26 |
| 11 | 4 | 78 | +2 | +4 | +16 | +46 | +28 | +32 |
| 12 | 5 | 217 | 0 | −2 | +1 | −13 | −23 | −23 |
| 13 | 5 | 217 | +2 | +4 | +8 | −53 | −50 | −39 |
| 14 | 6 | 96 | 0 | +10 | +10 | +25 | +8 | +31 |
| 15 | 6 | 96 | +6 | −9 | +22 | +11 | −21 | −29 |
| 16 | 7 | 133 | 0 | +6 | +4 | +17 | +29 | +49 |
| 17 | 8 | 73 | – | – | – | +20 | −1 | −9 |
| 18 | 8 | 73 | – | – | – | +52 | +55 | +17 |

</details>

### With vs without camera calibration

*Without* = raw image (not undistorted), fx = 12.5 mm / 5 µm = 2500 px (calibrated: 2554), principal point = sensor centre.

| method | H calib | H no calib | L calib | L no calib | W calib | W no calib |
|:---|---:|---:|---:|---:|---:|---:|
| DMC label | **3.1** | 6.0 | – | – | – | – |
| **v56-17 measured** (box + camera geometry) | 5.7 | 9.0 | **4.6** | 5.2 | 8.0 | 7.9 |
| v56-17 network sizes | 29.0 | 31.0 | 5.3 | 7.0 | 8.8 | 9.1 |
| v5only_3500-16 | 33.3 | 32.1 | 7.9 | 8.5 | 6.9 | 7.2 |
| v5only-18 | 36.6 | 35.2 | 8.7 | 8.9 | 7.3 | 7.8 |
| cam5_v4_e20-08 | 21.5 | 22.8 | 8.3 | 8.0 | 10.0 | 11.2 |

The DMC height needs the calibration. The two uncorrected effects partly cancel on this camera: datasheet fx alone
gives +10.8 cm, no undistortion alone −8.4 cm. The models hardly change (± 1.5 cm), because their height bias dominates.

### Overlays

Model v56-17. Left **with** calibration, right **without**. Green = the package box, orange = other detections,
cyan = DMC label; text: true size, network sizes (white), measured sizes (green) and DMC height with errors (cm).
Measured sizes on the box edges: L lower long edge, H upper long edge, W right short edge, true value in brackets.

![frame 4, package 2](assets/readme/overlay_00000004.jpg)
![frame 9, package 3](assets/readme/overlay_00000009.jpg)
![frame 14, package 6](assets/readme/overlay_00000014.jpg)
![frame 16, package 7](assets/readme/overlay_00000016.jpg)

<details>
<summary>All 19 frames</summary>

![all frames](assets/readme/overview.jpg)

</details>

### Our model vs production (`com.dachser.ilo.volume_measurement`)

Production run on the same 19 frames with its own detector code and weights (`tools/run_production.py`):
rotated-box YOLO `weights_oehringen.pt` on a 1200 × 1000 px ROI → L × W at the DMC scale, EU-pallet snapping,
YOLOv4-tiny pallet detector (→ 120 × 80), height from `calc_height`. Its ROI is normally placed from a *vehicle* code;
the test frames have none, so the ROI is centred on the package DMC (all crops contain the whole unit).

| | measured | L | W | H |
|:---|:---:|---:|---:|---:|
| **production** (on the frames it measured) | 11 / 19 | 9.9 | 6.8 | 11.1 |
| **ours measured** v56-17 box + camera geometry (same 11 frames) | 11 / 19 | **3.7** | 9.0 | **3.8** |
| **ours measured** v56-17 box + camera geometry (all frames) | **19 / 19** | 4.6 | 8.0 | 5.7 |
| our network sizes v56-17 (same 11 frames) | 11 / 19 | 4.3 | 8.8 | 24.5 |
| our network sizes v56-17 (all frames) | **19 / 19** | 5.3 | 8.8 | 29.0 |
| our network sizes v5only_3500-16 (same 11 frames) | 11 / 19 | 5.7 | **6.1** | 32.7 |

Mean absolute error in cm.

- Production measures only carton pallet loads: **no detection** on the dark board (pkg 3), the strapped machine (5),
  the overhanging panel (6, one of two frames) and the cage (7) — detector scores 0.0-0.23 on those crops; no DMC (8) →
  no measurement at all. No EU-pallet snapping and no pallet-detector override fired on these frames.
- Where it measures, its L is worse (frame 1: 142 for 120; frame 15: 99 for 160) and W similar; its height (DMC) is far
  better than our model's height, but 3× worse than our calibrated DMC height.
- Best combination: **our model for detection + 2D box, camera geometry for L × W, calibrated DMC for height** (measured).

<details>
<summary>Sizes per frame (cm)</summary>

| frame | pkg | true L × W × H | ours measured L × W × H | our network v56 L × W × H | production L × W × H |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 1 | 120 × 80 × 120 | 122 × 87 × 121 | 118 × 88 × 135 | 120 × 85 × 134 |
| 1 | 1 | 120 × 80 × 120 | 124 × 85 × 116 | 114 × 82 × 145 | 142 × 88 × 128 |
| 2 | 1 | 120 × 80 × 120 | 126 × 96 × 119 | 127 × 98 × 147 | 126 × 90 × 128 |
| 3 | 1 | 120 × 80 × 120 | 126 × 92 × 125 | 125 × 95 × 116 | 120 × 88 × 140 |
| 4 | 2 | 120 × 80 × 211 | 123 × 86 × 214 | 123 × 87 × 176 | 121 × 84 × 215 |
| 5 | 2 | 120 × 80 × 211 | 122 × 94 × 218 | 119 × 93 × 203 | 124 × 89 × 220 |
| 6 | 2 | 120 × 80 × 211 | 120 × 88 × 215 | 122 × 91 × 174 | 119 × 90 × 213 |
| 7 | 2 | 120 × 80 × 211 | 118 × 87 × 216 | 116 × 87 × 162 | 119 × 88 × 215 |
| 8 | 3 | 140 × 89 × 217 | 139 × 94 × 221 | 137 × 97 × 184 | *no package detected* |
| 9 | 3 | 140 × 89 × 217 | 142 × 99 × 220 | 141 × 101 × 133 | *no package detected* |
| 10 | 4 | 120 × 80 × 78 | 127 × 89 × 75 | 116 × 84 × 97 | 129 × 85 × 93 |
| 11 | 4 | 120 × 80 × 78 | 126 × 83 × 80 | 112 × 75 × 106 | 117 × 83 × 94 |
| 12 | 5 | 120 × 80 × 217 | 112 × 79 × 217 | 113 × 86 × 194 | *no package detected* |
| 13 | 5 | 120 × 80 × 217 | 104 × 90 × 219 | 117 × 96 × 167 | *no package detected* |
| 14 | 6 | 160 × 80 × 96 | 160 × 86 × 96 | 156 × 80 × 104 | *no package detected* |
| 15 | 6 | 160 × 80 × 96 | 158 × 68 × 102 | 165 × 74 × 75 | 99 × 76 × 118 |
| 16 | 7 | 159 × 115 × 133 | 145 × 104 × 133 | 128 × 91 × 162 | *no package detected* |
| 17 | 8 | 82 × 65 × 73 | 86 × 70 × 72 | 82 × 64 × 72 | *no DMC* |
| 18 | 8 | 82 × 65 × 73 | 80 × 60 × 128 | 78 × 61 × 128 | *no DMC* |

</details>

Left **our model** (green, calibrated image), right **production** (magenta, its raw input image). Sizes are written
on the box edges — L lower long edge, H upper long edge, W right short edge — with the true value in brackets (cm).

![frame 1: our model vs production](assets/readme/prod_00000001.jpg)
![frame 4: our model vs production](assets/readme/prod_00000004.jpg)
![frame 14: production detects nothing](assets/readme/prod_00000014.jpg)
![frame 15: production L 99 for 160](assets/readme/prod_00000015.jpg)

<details>
<summary>All 19 frames</summary>

![all frames, our model vs production](assets/readme/prod_overview.jpg)

</details>

### Box refinement: test-time augmentation + edge snapping

The 2D box was the weak part (W +4.9 cm on average: rotated a few degrees off the top face, or spilling over a wrapped
side onto the floor). Two inference-only steps in `tools/eval_real.py`, no retraining:

- `--tta`: the network on the 8 rotations / flips of the image, each box averaged over its matches. The 8 outputs
  disagree by up to ~10 network px on the centre, so the average is steadier.
- `--snap 15`: the two long edges (they set W) moved along their normal, within ±15 px, onto the strongest image edge.
  Only the long edges: snapping the short ones too pulls them onto carton seams inside the top (L 4.4 → 5.3).
  ±10 … ±20 px all give W 5.0-5.1, i.e. not tuned to these frames (an asymmetric −20 / +10 would give 4.1, not used).

| v56-17 measured, MAE cm | L | W | H | volume acc. |
|:---|---:|---:|---:|---:|
| box as predicted | 4.6 | 8.0 | 5.7 | 84.9 % |
| `--tta` | 4.4 | 7.0 | 5.7 | |
| **`--tta --snap 15`** | **4.4** | **5.1** | 5.7 | **88.7 %** |
| `--tta --snap 15`, without calibration | 4.9 | 4.9 | 9.0 | 89.9 % |
| **v567-13** (+ v7 data), box as predicted | 5.6 | 8.2 | 5.5 | 84.7 % |
| **v567-13 `--tta --snap 15`** | **4.4** | **4.7** | **5.5** | **90.1 %** |
| v567-13 `--tta --snap 15`, without calibration | 4.6 | 4.6 | 8.2 | 91.0 % |

v567 vs v56 (both `--tta --snap 15`, with calibration): W 5.1 → 4.7, H 5.7 → 5.5, volume 88.7 → 90.1 %. L/W better
on most frames; two worse: frame 13 (tall mixed load, W +2 → +10: the box now takes in more of the overhanging top)
and frame 16 (cage, W −5 → −14: one long edge ends up inside the top rail). H changes only through the DMC label box
(frame 17) and the network H of frame 18 (no label). The 19 frames hold few damaged loads, so they say little about what
v7 adds; the gain here is small (one or two frames' worth) — new real frames of damaged loads are needed to judge it.
The checkpoint from the last epoch (v567-19, `predictions/real_eval_v567last*`) gives L 4.0 W 5.6 H 5.6, volume 89.1 %.

All tables in `results.md` (generated by `tools/results_table.py`) use the refined v567 boxes (`real_eval_v567tta`).

### With calibration vs without vs production

`tools/overlay_calib_nocalib_production.py`: per frame three panels — v56-17 measured with calibration, v56-17 measured
without calibration, production. MAE in cm; production measures only 11 of the 19 frames (6 × no package detected,
2 × no DMC label). Accuracy = 100 % − mean |error| / true size; volume = L · W · H.

| | frames | L cm | W cm | H cm | L % | W % | H % | volume % |
|:---|:---:|---:|---:|---:|---:|---:|---:|---:|
| **v56-17 measured, with calibration** | 11 | **3.6** | 9.0 | **3.7** | **97.0** | 88.7 | **97.2** | 85.3 |
| v56-17 measured, without calibration | 11 | 3.8 | 8.8 | 6.3 | 96.9 | 89.0 | 94.4 | **86.4** |
| production (`volume_measurement`) | 11 | 9.9 | **6.8** | 11.1 | 92.9 | **91.5** | 89.7 | 77.0 |
| v56-17 measured, with calibration | 19 | 4.6 | 8.0 | 5.7 | 96.2 | 90.2 | 94.0 | 84.9 |
| v56-17 measured, without calibration | 19 | 5.2 | 7.9 | 9.0 | 95.8 | 90.3 | 90.7 | 85.9 |

Volume accuracy is limited by W (box too wide, +5 cm on average). Without calibration the volume comes out slightly
better only because its errors cancel: H is lower on some frames and offsets the too-large W.

Full-size images: `predictions/overlays_calib_vs_nocalib/`, `predictions/overlays_model_vs_production/`,
`predictions/overlays_calib_nocalib_production/`,
`predictions/overlays_combined/` (all three on one image: green = calib, yellow = no calib, magenta = production).
Tables as csv in `predictions/`: `height_error_cm.csv`, `height_error_cm_calib_vs_nocalib.csv`,
`height_error_cm_vs_production.csv`, `model_vs_production.csv`; production raw results: `predictions/production/`.

```bash
python3 tools/prepare_real_nocalib.py                                     # -> datasets/REAL_CAM_000CDF0AD90F_nocalib
python3 tools/eval_real.py --ckpt checkpoints/v56/v56-17.ckpt --out_dir predictions/real_eval_v56
python3 tools/eval_real.py --ckpt checkpoints/v56/v56-17.ckpt \
    --data datasets/REAL_CAM_000CDF0AD90F_nocalib --out_dir predictions/real_eval_v56_nocalib
python3 tools/overlay_calib_vs_nocalib.py --run real_eval_v56 --out_dir predictions/overlays_calib_vs_nocalib_v56

# production volume measurement: needs OpenCV 4 (Darknet pallet detector) and the repo's models/ + utils/ with a
# `polyiou` module on --code (the compiled extension, or a python stand-in computing polygon IoU)
venv_cv4/bin/python tools/run_production.py --repo ~/Downloads/com.dachser.ilo.volume_measurement-main --code <dir>
python3 tools/overlay_model_vs_production.py --run real_eval_v56 --out_dir predictions/overlays_model_vs_production_v56
python3 tools/overlay_calib_nocalib_production.py --run real_eval_v56   # -> predictions/overlays_calib_nocalib_production
python3 tools/eval_real.py --ckpt checkpoints/v56/v56-17.ckpt --tta --snap 15 --out_dir predictions/real_eval_v56tta
python3 tools/eval_real.py --ckpt checkpoints/v56/v56-17.ckpt --tta --snap 15 \
    --data datasets/REAL_CAM_000CDF0AD90F_nocalib --out_dir predictions/real_eval_v56tta_nocalib
for d in "" _nocalib; do
  python3 tools/eval_real.py --ckpt checkpoints/v567/v567-13.ckpt --tta --snap 15 \
      --data datasets/REAL_CAM_000CDF0AD90F$d --out_dir predictions/real_eval_v567tta$d
done
python3 tools/overlay_combined.py --run real_eval_v567tta               # -> predictions/overlays_combined
python3 tools/results_table.py --run real_eval_v567tta                  # -> results.md
```

## Commands

```bash
# regenerate / more data (CPU, low priority; does not touch the GPU)
nice -n 19 python3 synth/generate.py --n 3000 --workers 4 --seed 0
nice -n 19 python3 synth/generate.py --n 3000 --seed 1 --out datasets/SYNTH_CAM_000CDF0AD90F_s1

# real test set
python3 prepare_real.py

# look at labels
python3 tools/visualize_labels.py datasets/SYNTH_CAM_000CDF0AD90F /tmp/synth.jpg --n 16
python3 tools/visualize_labels.py datasets/REAL_CAM_000CDF0AD90F  /tmp/real.jpg  --n 19

# train (when the GPU is free) — warm-starts from custom_3d_model rtmdet_pack_v3, data = v4 + v3 by default
./train.sh --epochs 10
DATA="datasets/SYNTH_CAM_000CDF0AD90F_v4" RUN=cam5_v4only ./train.sh   # other data / run name

# evaluate on the 19 real frames: package box + L x W x H in cm next to the Readme sizes (csv, json, overlays)
python3 tools/eval_real.py --ckpt checkpoints/cam5_v4/<best>.ckpt   # -> predictions/real_eval/
#   rtmdet_pack_v3 as is (not trained on this camera):   MAE L 55.1  W 30.2  H 84.3 cm
#   cam5_v4-08.ckpt (10 epochs, v4 3000 + v3 505 images): MAE L  8.6  W 12.8  H 23.8 cm, 19/19 detected
#   cam5_v4_e20-08.ckpt (+10 epochs from it, --lr 2e-4):   MAE L  8.3  W 10.0  H 21.5 cm  (lr 1e-3 -> NaN)
INIT= ./train.sh --epochs 80          # from scratch

# predict on the real test frames (CPU)
cd ~/code_ws/study/3D_detection/custom_3d_model
CUDA_VISIBLE_DEVICES= python3 predict.py --ckpt <best.ckpt> \
  --folder ~/Documents/dachser/package_dims/datasets/REAL_CAM_000CDF0AD90F/shard_00000000 \
  --out_dir ~/Documents/dachser/package_dims/predictions \
  --intrinsics 1277.03 1277.39 632.45 504.42
```

## Compute cost (CPU)

Per-frame cost of the three measurement methods on the 19 real `cam_000CDF0AD90F` frames, **CPU only**
(`CUDA_VISIBLE_DEVICES=""`), Intel i7-13700H (20 logical CPUs), 1 warm-up pass + 2 timed passes. *Ours* =
`v567-13` in the `--tta --snap 15` config that produced the accuracy in `results.md` (`tools/eval_real.py`,
with the CPU tuning below); *production* = `com.dachser.ilo.volume_measurement` (`tools/run_production.py`).

| method | FPS (wall) | latency / frame | CPU time / frame | CPU usage (avg cores busy) | CPU usage (% of 20 CPUs) | peak RAM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Ours, **with** calibration        | 0.80 | 1.24 s |  7.5 CPU·s | 6.0 | 30 % | 1117 MB |
| Ours, **without** calibration     | 0.81 | 1.23 s |  7.4 CPU·s | 6.0 | 30 % | 1131 MB |
| Production (`volume_measurement`) | 3.60 | 0.28 s |  2.17 CPU·s | 7.8 | 39 % |  932 MB |

- **CPU tuning (on by default in `eval_real.py`, output unchanged):** (1) `channels_last` memory layout
  (faster oneDNN conv kernels); (2) `torch.set_num_threads(6)` — PyTorch defaults to 14 threads here, but
  spilling onto the E-cores is slower, so 6 performance-core threads wins; (3) `torch.compile` (graph fusion;
  all 8 TTA views are 640×640 so it compiles once, on the first frame). Together these took ours from
  **0.55 → 0.80 FPS** (1.45×) and cut CPU work from ~14 busy cores / 25.6 CPU·s to **6 cores / 7.5 CPU·s**
  per frame (≈3.4× less). All three only reorder float ops: sizes move ≤0.2 cm (vs the ~4 cm MAE) and no
  detection flips. Override with `--threads N` / `--no_channels_last` / `--no_compile`.
- **With vs without calibration cost the same** — identical network forward pass; only the camera
  intrinsics / geometry differ. The remaining cost is 8-fold test-time augmentation (8 forward passes per
  frame, ~80 % of the time) + long-edge snapping. **Without `--tta`** a single tuned pass runs at **~8 FPS**.
- **Production** runs a small rotated-box YOLO on a 416 px ROI, so it is ~4.5× faster per frame than our TTA
  pipeline — but our single tuned pass (~8 FPS) is ~2.2× faster than production. Its figures cover the 17
  frames whose detector ran (11 produced a measurement); the 2 no-DMC frames are skipped near-instantly.
- "avg cores busy" = CPU-seconds ÷ wall-seconds.

## Notes / known gaps

- Synthetic data is procedural, not photoreal: real tone is mean 66 / sigma 37 on valid pixels,
  synthetic 68 / 43 (3000-image set). Check with a few real frames after training;
  domain gap is the main risk.
- Height alternative that needs no learning: the DMC symbol is a fixed 94.8 mm, so
  `height = 5.702 - fx * 0.0948 / side_px` (undistorted px, full res) gave 3.1 cm MAE (max 7.4) on the real
  frames, see "Results on the real frames". `synth` json keeps `dmc_3d` so a DMC-based head/check can be trained or validated too.
- The label size 94.8 mm was fitted on the same 19 frames — measure a real label to confirm.
