# UmassLidar

A repo that builds Isaac Gym digital twins from UMass Amherst campus LiDAR data, and runs wheeled-robot
navigation experiments and driving-data collection inside them.

- **Main pipeline**: ground-collected session (LiDAR + photos + INS) → textured digital twin → LIMO waypoint navigation.
  The whole "RSS2027 Digital Twin" section below covers this.
- **Auxiliary tools**: early experimental code that crops a place out of airborne LiDAR (`Airborne.las`) and builds a 2.5D terrain. Last section.

# Datasets
## 1. 2026 03 07 snowy bluewall
Sensors: Inertial Labs RESEPI + Hesai 32ch LiDAR + Sony a5100
Data (https://drive.google.com/drive/u/1/folders/1MhfiWRgno-qihtz96PdWmms9i4fMCdGZ):
2026-03-07-18-07-32 (24 GB)	33 min run, 796 photos	PPK post-processing finished. Main session
2026-03-04-23-35-52 (1.5 GB)	20 s, 11 photos, lidar.scan 94 MB	raw only (no ppk.txt). A test capture.

```bash
camera/<serial>_<t0>_<t1>.jpg   796 photos, 6000×4000, Sony ILCE-5100
                                t is in µs, t/1e6 − 1e9 = LAS gps_time. EXIF orientation=3 (rotated 180°)
ins/ppk.txt                     PPK trajectory 125 Hz: Week GPSTime UTCTime Lon Lat H-Ell Heading Pitch Roll
ins/trajectories/ppk/*.txt      5 copies of the same trajectory differing only in geoid (WGS84 ellipsoid, GEOID18 etc.)
ins/SBET.OUT                    the same trajectory (SBET binary)
ins/*-imu.imr, *-gnss.gps etc.  raw IMU/GNSS and Waypoint Inertial Explorer intermediate files
data/hamp0660.*, igu*.sp3       PPK base station data (CORS 'hamp', RINEX obs/nav, precise ephemeris)
data/RESEPI-*.data (57 files, 15 GB) raw device logs
processed_ppk.pcmp              LiDAR lever arm, boresight, mount angle (pitch −45°), per-laser corrections
processed_ppk.pcpp              camera lever arm (0, −0.08, −0.048), boresight, mount −45°, photo list
log.txt                         collection summary: GNSS satellite count (21–25 for 70%), IMU statistics, range filter 8–200 m
```

## 2. Airborne.las EE2023
All-campus UAS (drone) aerial survey, Spring 2023. Sensors: DJI drone cameras only (Mavic 2 Pro, Mavic 3 Enterprise),
processed with Agisoft Metashape. **This is photogrammetry, not LiDAR**: `Airborne.las` is the dense point cloud
Metashape computed from the photos (RGB + normals, no intensity, no classification).

Two folders on Google Drive, both under `~/mydata/worldmodel/rss2027/EE2023/` on this machine, plus `~/Desktop/Airborne.las`.

**(a) Final products** — `AllCampus_Spring2023_Edits_v3/` (https://drive.google.com/drive/folders/1doCya4EbVbIss1Bv_Kyq4bfK6yQioaXi)

```bash
Point Cloud/AllCampusUAS_PointCloud_Spring2023.las   20.5 GB, 707.5 M points, Metashape export (2023-12)
    LAS 1.2 point format 2, UTM 18N metres, RGB + normals; extent 4.9 × 5.0 km, z −252 ... 404 m (outliers included)
~/Desktop/Airborne.las                               20.5 GB, 705.9 M points, 4.0 × 3.8 km, about 92 pts/m²
    the same cloud re-exported from QT Modeler (2025-07) with the outliers removed (z −3 ... 206 m). This is the
    default input of select_region.py / las_overview.py and the source of every "site" twin (pond, library, ...)
Ortho Tiles/*.tif                                    36 GB, 23 tiles (a16 ... e20), about 945 × 640 m each
    3-band orthomosaic, 0.1008 ft (≈ 3 cm) per pixel, EPSG:2249 (NAD83 Massachusetts Mainland, US survey feet)
    build_twin_site.py colours the ground from these automatically when the folder exists (umass_lidar/ortho.py)
Orthomosaic/AllCampusUASOrtho_Spring2023.tif         105 GB, the same orthomosaic as one file (not used)
Orthomosaic/Ortho with Pyramids/                     3 identical 51.7 GB copies of a lower-level export (not used)
DEM/, DEM Tiles/                                     elevation raster, 0.403 ft (≈ 12 cm) per pixel, EPSG:2249 (not used)
Tiling Footprint Masks/mask_<tile>.gpkg              tile outlines (tiles are buffered 50 ft beyond them)
READ_ME.txt                                          product list and the two Drive links
```

**(b) Raw survey and Metashape projects** — `Aerial Images/`, `AlignedChunksViaPointClouds/`
(https://drive.google.com/drive/folders/1kBTbf2Wi2AtV4I8o0_BCfMtxHIy65xED)

```bash
Aerial Images/RegionXX/DJI_*.JPG   25 regions, 12,514 photos
    mostly Mavic 2 Pro (5472×3648, no RTK, altitude about 122 m), vertical (nadir) capture
    Region17/18/19/23/24/25/26/WEB are oblique captures (gimbal −65°)
    only Region08_09_10_15_16 is Mavic 3E + PPK files (.nav/.obs/Timestamp)
AlignedChunksViaPointClouds/
    *_SubregionSet_1.psx/.files   chunk 17(R29), 18(R22), 38(R36_37_38_44), 40/41 merged
    *_SubregionSet_2_*.psx/.files chunk 10(R26), 14(R25), 15(R24), 16(R23), 37–39 merged
      each chunk: chunk.zip/doc.xml (aligned camera poses, calibration, chunk→ECEF transform)
                0/depth_maps (EXR, 1/8 of the original resolution), dense_cloud (proprietary .oc3)
                point_cloud (tie points), merged chunks also have elevation (DEM)
    transects.shp, Set2_Chunks*.png (screenshots before/after GCP alignment)
```

Which twin pipeline uses what:

| Pipeline | Input | Result |
|---|---|---|
| `select_region.py` → `build_twin_site.py` (Quick guideline) | `Airborne.las` + (a) ortho tiles | any campus area up to 25 ha; ground from the 3 cm ortho, structures from point colours |
| `build_twin_aerial.py` | (b) photos + Metashape poses of one chunk | photos projected on the mesh, including walls where oblique photos exist; only the processed regions (22–26, 29, 36_37_38_44) |

# Demo
A Clearpath Warthog and a ModalAI Starling 2 Max follow a fixed loop over the lawn next to the campus pond,
live in the Isaac Gym viewer. The window stays open afterwards so you can look around or drive the Warthog yourself.

Example files (https://drive.google.com/file/d/1IPwI7H2PyYNpbdXkeR2n2e6SCLDwieo6/view?usp=drive_link):
- **Pond twin**: `pond_twin.tar.gz` (913 MB, 1.4 GB unpacked; download link: TODO). Unpack in the repo root with
  `tar xzf pond_twin.tar.gz` → `assets/twin/pond/` and `configs/pond.json`.
- **Robots**: `python scripts/prepare_robots.py` (Warthog, processing env) and, optionally,
  `python scripts/prepare_robots.py --starling` (Starling meshes; without them a box model is used).
- **Isaac Gym Preview 4** env (section "Environments"), Linux + NVIDIA GPU.

```bash
conda activate umasstwin
python scripts/demo.py
```

| Key | Action |
|---|---|
| `1` / `2` | follow the Warthog / the Starling |
| `0` | free camera (mouse + WASD) |
| `←` `→` / `↑` `↓` | orbit / zoom |
| `T` | manual drive of the Warthog on / off (`I` `K` forward / back, `J` `L` turn) |
| `P` | pause |

- Loading takes 1–2 minutes before the window responds. Close the window or press Ctrl-C to stop.
- `python scripts/demo.py --record` also writes the videos to `outputs/demo/` (about 10× slower). Other arguments are passed
  on to `scripts/run_team.py`, e.g. `--speed 2 --follow uav --collision none`.
- If demo.py reports something missing it prints how to get it. Details on the robots and the viewer are in section 5d.


# Quick guideline
## 1. Select region.

```bash
conda activate umasstwin
# (first time only) generate the full overview map
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview
```
```bash
# draw a rectangle (or a polygon) on the map and crop that area to a LAS
python scripts/select_region.py --name pond --polygon
```

- Controls: left drag = rectangle (drag a corner to edit), scroll = zoom, right drag = pan, `Enter` = crop, `Esc` = cancel.
- **Polygon instead of a rectangle** (e.g. a skewed quadrilateral): press `p` (or start with `--polygon`), click the corners one by one,
  click the first corner again to close it, then drag any corner to adjust. `b` goes back to the rectangle, `Esc` clears the polygon,
  `q` quits. Only the points inside the polygon are written; the twin is still built on the polygon's bounding box, and the part
  outside the polygon becomes no-data (grey, not drivable).
- The title line shows size / area / estimated point count. Past the limits it turns red and `Enter` is refused, so redraw.
- Output: `data/sites/<name>.las` (same format and CRS as the source) + `<name>.json` (bbox, polygon corners if any, area, point count). About 30 s to scan the full 20 GB.
- Default limits: 500 m per side, 25 ha (estimated for an RTX 4090 24 GB). Change with `--max-side`, `--max-area`.
- For another LAS use `--las` and `--overview`; output location is `--out`.
- Without a display (remote), give coordinates directly: `python scripts/select_region.py --bbox XMIN YMIN XMAX YMAX --name b1`,
  or a polygon with `--poly X0 Y0 X1 Y1 X2 Y2 ...` (UTM, at least 3 corners).

## 2. Build twin from the region

```bash
conda activate umasstwin
python scripts/build_twin_site.py --las data/sites/pond.las      # library 9.9 ha: a few minutes
python scripts/build_collision.py --config configs/pond.json                # mesh collision for buildings / trees / cars (tens of seconds, also the reference for voxel)
python scripts/build_collision.py --config configs/pond.json --mode voxel   # voxel collision = recommended default (about 2 min, section 6)
```


- The input is point positions and colours (no photos or sensor trajectory needed), plus the EE2023 orthomosaic tiles for the ground
  when `~/mydata/worldmodel/rss2027/EE2023/AllCampus_Spring2023_Edits_v3/Ortho Tiles` exists: the ground is then textured at 3 cm from
  the ortho wherever it is visible from above (parking lines, cracks and grass texture become readable), and from the points elsewhere.
  `--ortho DIR` uses another folder, `--no-ortho` turns it off. The first run creates `configs/library.json`, and every script uses this config afterwards.
- Output: `assets/twin/library/` (`ground.glb` collision mesh, `map.png` map for waypoints, `full/` tile GLB / PNG / `twin.json`); the cache is `data/library/`.
- Per-stage cache: running again only does the export (about 10 s). To recompute, use `--redo ground|cloud|ortho|bake`.
- A quick partial build: `--region X0 Y0 X1 Y1` (local m, `map.png` coordinates) → saved separately as the `crop` variant.
- Quality options (changing them needs `--redo bake`):
  `--tex-res` ground texel (default 0.03 with the ortho, 0.05 without: point spacing is about 0.1 m, so going lower does not make it sharper), `--smooth 0.07` point colour blur width,
  `--fill 0.6` width for filling empty ground (under trees etc.) with surrounding colour, `--voxel 0.12` structure mesh density (too small fragments, too large blurs),
  `--min-h 0.4` only points above this height become 3D structure.
- Green in `map.png` = drivable. Buildings, trees and parked cars are detected as obstacles.

## 2b. Correct the drivable area by hand (optional)

Where a UGV may drive is decided automatically: the semantic grid (ground / obstacle / roof / no data), a slope limit, and the
robot's footprint. That is right for buildings, trees and steep ground, but it can cut an area off behind one narrow passage
(a few stray obstacle or slope cells), keep objects that were only there on the survey day, and it knows nothing about where
an experiment should not go. `edit_drivable.py` lays hand-drawn polygons over the automatic result.

```bash
conda activate umasstwin
python scripts/edit_drivable.py --config configs/pond.json
```

| Input | Action |
|---|---|
| `A` / `B` | start an **allow** polygon (cells inside become drivable) / a **block** polygon (cells inside are forbidden) |
| left click | add a corner; right click / `Backspace` removes the last one |
| `Enter` | finish the polygon: it is saved at once and the colours are recomputed |
| `D`, then click inside a polygon | delete that polygon |
| `Esc` / `Q` | drop the polygon being drawn / quit |

- Colours: red = not drivable, orange = drivable but cut off from the largest connected area. The title shows both areas in hectares,
  so a successful connection is visible as the largest area growing and the orange disappearing.
- The edits are stored in `assets/twin/<name>/drivable_edits.json` (polygons in local metres, applied in order). The automatic
  grids are never modified: delete the file (or `--clear`) to go back, and a rebuilt twin keeps the edits.
- They are used wherever a UGV route is planned (`run_fleet.py` including its waypoint window, `run_nav.py`, `run_team.py`).
  UAVs ignore them.
- **Allow only changes where routes may go, not the collision ground.** Obstacle cells are raised by up to 0.6 m in the ground
  mesh and 3D structures keep their collision, so a route through an allowed area can still hit something real. The window
  (and `--allow`) warns when a polygon covers raised cells. Flattening the ground there is not implemented.
- Block is complete on its own: routes never enter a blocked polygon.
- The preview uses the Warthog footprint (`--robot`, `--max-slope` change it); the planner always uses each robot's own.
- Without a display: `--allow X0 Y0 X1 Y1 X2 Y2 ...`, `--block ...`, `--list`, `--clear`.

**Checked on the pond twin.** The parking lot in the south-west (1.7 ha) is drivable but not connected to the main area for a
Warthog: the two meet in an alley between buildings at about (139, 268), 1.5 m apart. An allow rectangle over that alley
(136–143, 264–273) raised the largest connected area from 12.77 to 14.46 ha, and a route from the lawn to the parking lot
(638 m) was planned; without the edit the same route is refused. That rectangle covers 9 m² of raised cells, so the drive
itself was not tested and may bump there. Only the command-line form and simulated clicks were tested, not the real window.


## 3. Prepare the robots (once)

Every navigation script uses the same two robots by default: the UGV is a **Clearpath Warthog**, the UAV is a
**ModalAI Starling 2 Max**. Their models are not stored in this repo; one script downloads and converts them.

```bash
conda activate umasstwin
python scripts/prepare_robots.py              # Warthog (also Jackal, Husky) -> assets/robots/<name>/ + robots.json
python scripts/prepare_robots.py --starling   # Starling 2 Max meshes -> assets/robots/starling/meshes/
```

| Robot | Where the model comes from | What the script does |
|---|---|---|
| Warthog (UGV) | official [`warthog_description`](https://github.com/warthog-cpr/warthog) (BSD, branch `kinetic-devel`), cloned into `third_party/` | expands the xacro to a plain URDF, converts the meshes for Isaac Gym, drops the fender / light collision meshes, and writes the wheel names, track (1.14 m) and wheel radius (0.3 m) to `assets/robots/robots.json` |
| Starling 2 Max (UAV) | ModalAI publishes no URDF. Mass (566 g), motor layout (260 × 190 mm) and prop size come from the [datasheet](https://docs.modalai.com/starling-2-max-datasheet/); body / propeller meshes and inertia from the community model [AIRLAB_Starling2Max](https://github.com/BjarkeHJ/AIRLAB_Starling2Max) | clones that repo into `third_party/` and writes the two meshes as OBJ in metres. The URDF itself is generated at run time by `umass_lidar/uav.py` (`STARLING` parameters) |

- Both `third_party/` and the generated `assets/robots/` folders are gitignored, so run this once on every machine.
- The Starling source repo has no licence file, which is why its meshes are fetched locally instead of being committed.
  Without them the Starling falls back to a box-and-disc body with the same dynamics.
- Model details, estimated parameters and known limits are in section 5d.

## 4. Run navigation 

### teleoperation (single UAV or single UGV)
`scripts/run_teleop.py` puts one robot in a twin and lets you drive or fly it with the keyboard. Select the robot type with
`--platform ugv` (Warthog) or `--platform uav` (Starling 2 Max). The views are the same as in `run_fleet.py`
(third-person follow, fixed top view, free camera, trajectory line).

```bash
conda activate umasstwin
# drive a Warthog: click the start on the map, press Enter, then the sim opens
python scripts/run_teleop.py --config configs/pond.json --platform ugv --collision voxel
# fly a Starling from a given start (local metres), no map window
python scripts/run_teleop.py --config configs/pond.json --platform uav --start 340 450 --collision voxel
```

**Start.** Without `--start X Y` a window shows the twin's top view: click where the robot starts and press `Enter`
(for a UGV, red = not drivable; a click there is moved to the nearest drivable ground within 20 m). `--yaw DEG` sets the
heading (0 = east, 90 = north).

**Teleop keys.** Click the Isaac Gym window once so it has the keyboard focus.

| Key | UGV | UAV |
|---|---|---|
| `I` / `K` | forward / back | forward / back along its heading |
| `J` / `L` | turn left / right | yaw left / right |
| `U` / `O` | - | strafe left / right |
| `Y` / `H` | - | climb / descend |
| `=` / `-` | speed ×1.25 / ×0.8 | same |

- With no key held the UGV stops and the UAV holds its position. The UAV takes off to `--agl` (3 m) by itself at the start.
- Top speed is `--speed` (default 3 m/s), turn rate `--turn` (1.2 rad/s), UAV vertical speed `--climb` (2 m/s).
- There is no obstacle avoidance or geofence: the robot collides with whatever `--collision {mesh,voxel,none}` provides (section 6).

**View keys.** `1` follow the robot (default at start; `--view top` / `--view free` change it), `M` fixed top view of the whole
map, `0` free camera, `N` trajectory on / off, `←` `→` orbit, `↑` `↓` zoom, `P` pause. Close the window or press Ctrl-C to stop.

**Output** (`--out`, default `outputs/teleop/<twin>_<platform>_<time>/`): `trajectory.csv` (30 Hz pose, UTM and the key
command), `trajectory.png` (the path on the map), `result.json` (distance, duration, start).

**Verification (pond twin, voxel collision).** With the key input replaced by a constant command for 25 s: the Starling took off,
flew 69 m on a climbing curve (z 6 → 22 m); the Warthog drove 75 m on a curve. The follow view and trajectory line were checked on
a screenshot. Real key presses and the start-selection window were not tested (the window shares its code with `run_fleet.py`).

### manual waypoint navigation (single ~ multi UAV-UGV)
`scripts/run_fleet.py` is the main experiment script: any number of UGVs and UAVs in one twin, each with its own
waypoints, watched live with every robot's trajectory drawn in its own colour.

```bash
conda activate umasstwin
# 2 Warthogs + 1 Starling in the pond twin: pick the waypoints on the map, then the sim opens
python scripts/run_fleet.py --config configs/pond.json --ugv 2 --uav 1 --speed 3 --collision voxel
```

**Settings.** `--ugv N` and `--uav N` set how many of each (default 1 and 1; 0 is allowed). The robots are named
`ugv1, ugv2, ..., uav1, ...`. The models are always Warthog and Starling unless `--ugv-model` / `--uav-model` say otherwise.

**Modes (`--mode`).**

| Mode | What happens |
|---|---|
| `waypoint_select` (default) | A window shows the twin's top view. Click the waypoints of each robot in turn, then the Isaac Gym sim starts with those routes. |
| `file` | Skips the window and reads the routes from `--routes routes.json`. Every run writes its `routes.json` into the output folder, so any run can be repeated. |
| `algorithm` | **Future work.** Reserved for the experiment algorithm that will drive the robots instead of fixed waypoints. It currently stops with a message. The interface kept for it is `AlgorithmPolicy` in `run_fleet.py`: `reset(world, robots)` and `step(t, robots) -> {name: command}` with a UGV command `(v, w)` and a UAV command `(velocity xyz, yaw)`; the waypoint follower is the same kind of object (`WaypointPolicy`). |

**Waypoint window.**

| Input | Action |
|---|---|
| left click | add a waypoint for the current robot (the first click is its start) |
| right click / `Backspace` | remove the last waypoint |
| `R` | clear the current robot's waypoints |
| `Enter` / `Space` / the green button | plan this robot's route and move on to the next robot; after the last robot the sim starts |
| `Q` or closing the window | cancel |

- The title line names the robot being edited. `Undo` and `Clear` are also buttons under the map.
- **Red = not drivable** for the UGV being edited (obstacles, roofs, slopes over `--max-slope`, no data, and a margin of half
  the robot width + 0.35 m around all of those). **Orange** appears once the start is placed: drivable ground that has no
  connection to that start wide enough for the robot. Neither can be used as a waypoint. To change what is red or orange,
  see section 2b.
- A UGV route is an A* ground path through the clicked points (points on blocked ground snap to the nearest drivable cell, up to 20 m).
  A click in the red / orange within 20 m of reachable ground is accepted and snapped. If a waypoint cannot be reached, a red message
  on the map says which one, the window stays on that robot, and the route stays editable.
- A UAV takes off at its first point and flies straight legs at `max(ground + --agl (5 m), structure top + --clearance (2 m))`,
  then hovers at the last point.
- The toolbar's zoom / pan tools work; clicks are ignored while one of them is active.

**Viewer keys.**

| Key | Action |
|---|---|
| `1` … `9` (top row or numpad) | follow that robot in third person, in the order `ugv1, ugv2, ..., uav1, ...` |
| `M` | fixed top view of the whole map, north up (the default at start; `--view ugv1` etc. changes it) |
| `0` | free camera (normal Isaac Gym mouse / WASD control) |
| `N` | trajectories on / off |
| `←` `→` / `↑` `↓` | orbit by 15° / zoom |
| `T` | manual drive of the followed UGV on / off (`I` `K` forward / back, `J` `L` turn) |
| `P` | pause |

- Each robot's travelled path is drawn as a line in its own colour in every view. In the top view the lines are drawn above
  the buildings and trees, so they are never hidden, and a ring marks each robot's current position.
- `--hold` keeps the window open after every robot has finished. `--headless` runs without a window.

**Output** (`--out`, default `outputs/fleet/<twin>_<time>/`):

| File | Content |
|---|---|
| `routes.json` | robots and their waypoints (local metres); replay with `--mode file --routes` |
| `plan.png` | planned routes on the map |
| `trajectories.png` | travelled paths on the map, one colour per robot |
| `trajectory_<robot>.csv` | per frame (30 Hz): pose, yaw, UTM, target waypoint |
| `result.json` | per robot: status (`reached` / `timeout` / `stuck` / `flipped` / `crashed`), distance, waypoint times, goal error |
| `topview.mp4` | only with `--record`: fixed top view of the whole map with the trajectories |
| `<robot>_rgb.mp4`, `<robot>_depth.mp4` | only with `--record-ego`: D455-like camera on every robot (slow) |

- Other options: `--speed` (UGV, default 1 m/s; the Warthog runs here used 3), `--uav-speed` (default 3 m/s), `--agl`, `--clearance`,
  `--collision {mesh,voxel,none}` (section 6), `--max-seconds` (default 900), `--plan-only` (stop after `plan.png`).
- Robots collide with each other. Routes are planned per robot with no coordination, so crossing routes can bump.
- Loading the pond twin takes 1–2 minutes and about 33 GB of RAM before the window responds.

**Verification (pond twin, voxel collision, `--speed 3 --uav-speed 5`).**
- Headless, 2 Warthogs + 2 Starlings, 150 s limit, `--record`: both Starlings reached every waypoint (492 m and 390 m,
  goal error ≤ 0.01 m). Neither Warthog finished in 150 s (146 of 190 m and 210 of 339 m). That run used a 0.8 m pure-pursuit
  lookahead, which made the Warthogs weave; the lookahead now grows with speed (2.4 m at 3 m/s).
- With the viewer, 1 Warthog + 1 Starling, 45 s: the Warthog reached its first waypoint (95 m) at 38 s instead of 139 s with the
  longer lookahead. Top view, follow view and trajectory lines were checked on screenshots. A full mission with the new
  lookahead has not been run, and neither have the keys, the waypoint window on a real display (only simulated clicks) and robot-robot contact.
- With the viewer the pond twin runs at roughly 0.4× real time.

The scripts below are the earlier single-purpose experiments. They still work and record more camera views.


## 5. Compare texture using Blender

```bash
# Without a driving log: place the D455 at LIMO camera height at map coordinates (x y yaw°). yaw 0 = east, 90 = north
blender --python scripts/blender_twin.py -- --twin assets/twin/library/full --at 170 150 90

# Several viewpoints as stills without a window (frame 0, 1, ... = order of --at)
blender -b --python scripts/blender_twin.py -- --twin assets/twin/library/full \
    --at 170 150 90 --at 150 135 0 --render outputs/blender/library --frames 0 2 1

# With a driving log, follow that trajectory
blender --python scripts/blender_twin.py -- --twin assets/twin/library/full --run outputs/nav/library_000
```

- `Numpad 0` = D455 view; viewport shading is Material Preview. Leave colour management at `Standard`.
- Colour adjustment: in the Shader editor, changing `Gain` / `Saturation` / `Gamma` inside the `TwinShade` group applies to all tiles immediately.
  Write the chosen values into `configs/library.json` as `"grade": {"gain": 1.2, "saturation": 1.1, "gamma": 0.95}` and rerun the commands of step 2
  to bake them into the textures. After that, set the knobs back to 1 in Blender and run `for im in bpy.data.images: im.reload()`.
- Checking the source: set `ShowSource` to 1. Blue = point colour, light blue = filled with surrounding colour, orange = no colour information (material mean colour).
- When mesh / resolution options change, bake a small area with `--region` and render `--twin assets/twin/library/crop` with the same `--at` to compare.
- Full description and parameter table: [docs/blender_tuning.md](docs/blender_tuning.md)

--- below is modyfying.

## 5a. Single UGV (`run_nav.py`)

```bash
conda activate umasstwin
# structure collision: voxel (recommended default, collision with holes filled, structure_collision_voxel.npz)
python scripts/run_nav.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
    --out outputs/nav/library_000 --headless --collision voxel
# structure collision: mesh (collision simplified from the visual mesh. This is what you get if --collision is omitted)
python scripts/run_nav.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
    --out outputs/nav/library_000_mesh --headless --collision mesh
# no structure collision (structures are visual only, collision is ground + 0.6 m stubs)
python scripts/run_nav.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
    --out outputs/nav/library_000_none --headless --collision none
```

- `--collision {mesh,voxel,none}`: the visual mesh (`s_*.glb`) is the same in all three cases and only the collision shape changes (section 6). Same for `run_uav.py`.

- Waypoints are local m read from `assets/twin/library/map.png` (the first point is the start). `--plan-only` lets you check the path first.

## 5b. Single UAV (`run_uav.py`)

```bash
conda activate umasstwin
# structure collision: voxel (recommended default)
python scripts/run_uav.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/uav/run_000 --headless --collision voxel    # same for other twins, e.g. --config configs/library.json
# no structure collision (for mesh collision use --collision mesh or omit it)
python scripts/run_uav.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/uav/run_000_nocol --headless --collision none
# collision test: fly low, ignoring structure height (when ON, it is blocked by the wall of the box structure)
python scripts/run_uav.py --waypoints 104 171 130 171 --agl 1.5 --clearance -100 --speed 2 --max-seconds 40 \
    --out outputs/uav/box_on --headless
```

- Takes off at the first waypoint and flies each leg at a constant altitude: `max(ground + --agl (6 m), structure surface + --clearance (2 m))`.
  The structure surface is the height obtained by rasterizing the twin's `s_*.glb` tiles (`<variant>/structure_top.npy`, computed once and cached), widened by 1.5 m.
- Going below the structure height is recorded as `structure_hit`; an actual collision is recorded as `contact_steps` / `first_contact`
  (actual collisions only when the `structure_collision.npz` of section 6 exists).
- Camera: the same RGB-D as the D455, mounted under the nose, `--cam-pitch-deg` (default 30° down), `--depth-max` (default 20 m).
- Output is the same as `run_nav.py` (`combined.mp4`, `d455_rgb/depth.mp4`, `depth/*.png` uint16 mm, `trajectory.csv` with rotor thrust and AGL added, `result.json`).
- Vehicle model (`umass_lidar/uav.py`): `--uav starling` (default, section 5d) or `--uav quad`, a 1 kg X-quad with 4 rotors of at most 6 N each
  (thrust/weight about 2.4). Both use a first-order motor lag of 30 ms and linear drag.
  Isaac Gym has no rotor model, so every step the summed force and torque of the 4 rotor thrusts are applied directly to the body.
  There are three levels of input usable as an RL action: `Quadrotor.apply_motors(f)` (rotor thrusts),
  `QuadController.ctbr(T, rates)` (collective thrust + body rate), `QuadController.velocity(v, yaw)` (velocity command, what waypoint flight uses).

**Verification (`outputs/uav/run_000`, blue_wall, `--uav quad`)**: all 3 waypoints reached (takeoff 4.1 s, 12.0 / 22.0 / 28.6 s), planned 60.8 m / flown 60.7 m,
goal error 0.11 m, maximum path deviation 0.29 m, 857 frames, 77% valid depth pixels.

## 5c. One UGV + one UAV on the same path (`run_team.py`)

```bash
conda activate umasstwin
python scripts/run_team.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
    --out outputs/team/library_voxel --headless --collision voxel     # or --collision mesh / none
python scripts/run_team.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/team/blue_wall_voxel --headless --collision voxel   # snowy bluewall (configs/blue_wall.json)
python scripts/run_team.py --config configs/blue_wall_aerial.json --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/team/blue_wall_aerial_voxel --headless --collision voxel
```

- The UGV and the UAV follow the same waypoints along the same A* ground path in one sim. The UAV flies above that path
  at height `max(ground + --agl (5 m), structure surface + --clearance (2 m))` (smoothly interpolated along the path).
- The default `--uav-speed` equals the UGV speed (they move side by side). The UAV takes off 1.5 m behind the UGV start point.
- Output: `combined.mp4` (map seen from above + third person + D455 RGB/depth of both robots), `topview.mp4`, `chase.mp4`,
  `ugv_rgb/depth.mp4`, `uav_rgb/depth.mp4`, `depth_ugv/`, `depth_uav/`, `trajectory_ugv.csv`, `trajectory_uav.csv`, `result.json`.

## 5d. Robot details and the live viewer

The default UGV is the **Clearpath Warthog** and the default UAV is the **ModalAI Starling 2 Max**
(`--robot warthog`, `--uav starling`). The older results in this README (LIMO, 1 kg generic quad) were produced
with `--robot limo --uav quad`; pass those to reproduce them.

| Robot | Flag | Model source | Notes |
|---|---|---|---|
| Warthog | `--robot warthog` (default) | official `warthog_description` URDF and meshes | 260 kg, track 1.14 m, wheel radius 0.3 m. Fender / light collision meshes are dropped (Isaac Gym turns them into convex hulls that drag on the ground). The two rocker joints are held by a velocity drive, not a real differential. |
| LIMO / Jackal / Husky | `--robot limo` etc. | AgileX / Clearpath descriptions | as before |
| Starling 2 Max | `--uav starling` (default) | mass 566 g, 260 × 190 mm motor layout, 180 mm props from the [datasheet](https://docs.modalai.com/starling-2-max-datasheet/); body / prop meshes and inertia from the community model [AIRLAB_Starling2Max](https://github.com/BjarkeHJ/AIRLAB_Starling2Max) | ModalAI publishes no URDF. **Max thrust (5.2 N per rotor), motor constants and drag are estimates, not measurements.** The camera is still the D455-like RGB-D, not the Starling's own tracking / hi-res / ToF cameras. Propellers are static visuals. |
| Generic quad | `--uav quad` | parametric box-and-disc model | 1 kg, 6 N per rotor |

```bash
conda activate umasstwin
python scripts/prepare_robots.py              # Warthog, Jackal, Husky (+ LIMO with --limo-desc)
python scripts/prepare_robots.py --starling   # Starling 2 Max meshes -> assets/robots/starling/meshes/
```

- The Starling meshes are **not in this repo**: the source repo has no licence file, so they are fetched locally
  (`assets/robots/starling/` is gitignored). Without them `--uav starling` falls back to a box-and-disc body with the same dynamics.
- `--speed` defaults to the robot's top speed capped at 1.0 m/s. The Warthog runs below used `--speed 3`.
- Skid-steer robots yaw much slower than the wheel differential implies in this sim (commanded 1.5 rad/s gave about 0.15 rad/s
  for both the Warthog and the LIMO on flat ground). The Warthog has an experimental yaw-rate PI loop (`yaw_feedback` in
  `assets/robots/robots.json`, 1.3 rad/s on flat ground); it has not been shown to shorten a full mission, set it to `false` to turn it off.

**Live viewer.** Without `--headless` a viewer window opens and its camera follows the robot in third person.
Clicking an actor in the Isaac Gym side panel cannot be read from Python, so the target is chosen with keys:

| Key | Action |
|---|---|
| `1` / `2` | follow the UGV / the UAV (`run_team.py`; `run_nav.py` and `run_uav.py` have one target, key `1`) |
| `0` | free camera (normal Isaac Gym mouse / WASD control) |
| `←` `→` / `↑` `↓` | orbit by 15° / zoom in, out |
| `P` | pause |

```bash
conda activate umasstwin
# Warthog + Starling in the pond twin, live, following the Warthog
python scripts/run_team.py --config configs/pond.json --waypoints 160 300 185 400 140 460 135 330 \
    --speed 3 --collision voxel --no-record --out outputs/team/pond_live
```

- `run_team.py --follow {ugv,uav,free}` sets the camera at start; `run_nav.py` / `run_uav.py` use `--free-camera` to start free.
- `run_team.py --no-record` skips the five recording cameras and the videos (only the CSVs and `result.json` are written).
  Recording makes the sim run roughly 10× slower than real time on the pond twin, so use it for live viewing.
- Loading the pond twin (19.2 M visual faces + voxel collision) takes 1–2 minutes with no output before the window responds.

**Verification (pond twin, voxel collision).** Warthog + box-model Starling, `--speed 3`, before the yaw-rate loop
(`outputs/team/pond_warthog_starling`): both reached all 3 waypoints; Warthog 412 m in 377 s on a 322 m path (it loses time
turning at waypoints), goal error 0.37 m; Starling 328 m, 0 contacts. Starling with the mesh model alone
(`outputs/uav/pond_starling_mesh`, `--speed 4`): 2 waypoints, 89.0 m in 35.5 s, goal error 0.09 m, max cross-track 1.15 m.
The Warthog + mesh-model Starling combination and the viewer keys have only been smoke-tested (a few seconds, no errors).

## 6. Structure collision (mesh collision for buildings / trees / cars)

```bash
conda activate umasstwin
# two lines per twin: mesh mode (a few seconds) → voxel mode (1–2 min). At run time use --collision voxel (recommended default)
python scripts/build_collision.py                                              # blue_wall
python scripts/build_collision.py --mode voxel
python scripts/build_collision.py --config configs/blue_wall_aerial.json
python scripts/build_collision.py --config configs/blue_wall_aerial.json --mode voxel
python scripts/build_collision.py --config configs/library.json
python scripts/build_collision.py --config configs/library.json --mode voxel
python scripts/build_collision.py --config configs/library.json --variant crop   # when using the crop variant (same for --mode voxel)
```

**Why voxel is the recommended default (summary)**

- Problem: the mesh collision is a simplified copy of the visual mesh. The visual mesh cuts away "faces far from points" after Poisson reconstruction,
  so holes appear on walls and lower parts of buildings where LiDAR / photo points are sparse, and those holes stay in the collision as they are (a robot can pass through a wall).
- How voxel works: without going through the visual mesh, the point cloud is filled directly into a 12.5 cm lattice (a cell is blocked if it holds even one point),
  gaps of about 0.75 m or less are filled by morphological closing (dilation then erosion), and the outer surface of that blob is used as the collision mesh.
  There is no step that "estimates and connects" a surface, so even sparse places are blocked as far as there are points, and the result is always a closed shell.
- Why it is only a partial fix: holes larger than 0.75 m (places with no points at all, e.g. a wall under eaves in a point cloud captured only from above) remain.
- Numeric example (library, building tile 280–300 / 220–240 m): among points in the robot height band (0.4–2.5 m above ground), the fraction within
  0.25 m of the collision surface goes from mesh 28.9% → voxel 95.4%, and the hole boundary length (open edge) from 299 m → 0 m.

- The 3D structure tiles (`s_*.glb`) are originally visual only. This script turns all tiles into one static collision mesh
  (`<variant>/structure_collision.npz`). Processing: keep only the outward one of the double-sided faces, remove small pieces (fewer than 60 faces or smaller than 0.4 m),
  quadric decimation to 12% of the faces (`fast_simplification`), and push in by 6 cm against the surface normal.
  Isaac Gym draws collision meshes in grey, so pushing them inward hides them behind the textured surface.
- `run_nav.py` / `run_uav.py` turn collision on automatically when this file exists (`--collision mesh`, the default). To turn it off use `--collision none`
  (`--no-structure-collision` is kept with the same meaning).
- `run_uav.py` detects collisions by contact force and records them in `result.json` as `contact_steps`, `first_contact`, `blocked`.
- blue_wall: 4.13 M → 0.53 M faces, 8 s. Pixels where the grey collision mesh showed through in the D455 video were below 0.1%
  (comparing ON/OFF frames of the same route).

### Voxel collision (`--mode voxel`): collision without holes

```bash
conda activate umasstwin
python scripts/build_collision.py --config configs/library.json                 # mesh mode first (the reference for hiding voxel faces)
python scripts/build_collision.py --config configs/library.json --mode voxel    # → <variant>/structure_collision_voxel.npz
# usage: run_nav.py / run_uav.py --collision voxel
```

- Why: mesh mode simplifies the visual mesh as is, so holes in the visual mesh caused by sparse points stay in the collision.
  Voxel mode does not use the visual mesh and builds directly from the point cloud. The visual mesh (`s_*.glb`) and the mesh-mode file are not changed;
  the two collision files sit side by side and you pick one at run time with `--collision {mesh,voxel,none}`.
- Processing: points at least `--min-h` (0.3 m) above ground → occupancy on a `--voxel` (0.125 m) lattice → morphological closing (`--close` 0.375 m,
  fills gaps of about 0.75 m or less) → remove detached pieces smaller than `--min-voxels` (12) → voxel boundary faces → decimation to 12% of the faces.
- Input point cloud: `data/<name>/cloud.npz` (aerial / site twin) or `data/<name>/tiles/*.vox` (LiDAR twin). Confirmed working on all three twins.
- Hiding faces (default, turn off with `--no-hide`): the voxel shell lies up to 1 voxel outside the real surface, so it shows as grey in Isaac Gym.
  Vertices in front of the mesh-mode surface are moved onto that surface. Places without a visual mesh (holes) are left as they are, so holes are plugged by grey faces.
- Limits: holes larger than 0.75 m remain (walls under eaves in a point cloud seen only from above, etc.). Real gaps narrower than 0.75 m are also closed.
  Small isolated point clusters (fewer than 12 voxels) are dropped from the collision. The face count is about 5 times that of mesh mode.

**Test (library, three 20 m tiles with many holes)**: "fraction of points more than 0.25 m away from the collision surface" (lower is better) and hole boundary length.

| Region (local m) | Content | Uncovered points, all (mesh → voxel) | Uncovered points, 0.4–2.5 m above ground | Hole boundary (open edge) |
|---|---|---|---|---|
| A: 100–140, 20–40 | low, sparse points | 61.0% → 25.6% | 60.1% → 24.9% | 287 m → 0 m |
| B: 280–300, 140–160 | tall building | 11.5% → 0.6% | 64.9% → 3.8% | 581 m → 0 m |
| C: 280–300, 220–240 | building | 14.0% → 0.5% | 71.1% → 4.6% | 299 m → 0 m |

- The 25% left in A are small isolated point clusters filtered out by `--min-voxels` (`--min-voxels 1` keeps them, but noise comes in too).
- Whole library: 12.3 M points → 7.57 M voxels (2.78 M of them filled by closing) → 3.89 M faces, 100 s.
- Driving the same route (141.5 m) with `run_nav.py`: `none` / `mesh` / `voxel` all reach the goal with identical distance and time (the route does not touch structures).
  Pixels where D455 depth became nearer than `none`: mesh 0.11%, voxel 1.20% (of which more than 0.3 m nearer = plugged holes 0.70%,
  0.3 m or less = slight protrusion in front of the visual surface 0.50%). Before hiding faces 15.3% of RGB pixels changed; after hiding, 3.3% (mesh 0.3%).

**Examples** (UAV deliberately sent below structure height: `--clearance -100`):

| Example | Command | collision ON | OFF |
|---|---|---|---|
| Tree canopy (3 m above ground) | `--waypoints 106 180 106 218 --agl 3` | contact at (106, 195.1) at 11.1 s → flips and crashes | passes through the tree and reaches the goal |
| 3 m high box structure (1.5 m above ground) | `--waypoints 104 171 130 171 --agl 1.5` | stopped by the wall at x=114.8 (BLOCKED) | passes through |

Comparison videos: `outputs/uav/tree_compare.mp4`, `outputs/uav/box_compare.mp4` (left ON, right OFF, third person).


---

# RSS2027 Digital Twin (main pipeline)

Takes one session collected on foot with a RESEPI payload as input, builds a photo-textured 3D environment, and
drives one LIMO through waypoints in Isaac Gym while recording three views: **top view / third person / RealSense D455 RGB-D**.

```
RESEPI session                         build_twin.py                                  run_nav.py            (env: umasstwin)
┌───────────────────────┐   prep    ┌──────────────────────────────────────┐      ┌─────────────────────────┐
│ LAS  (LiDAR, 443 M pt)│──────────▶│ LAS↔trajectory shift, tile/voxel cache│      │ Isaac Gym               │
│ camera/*.jpg (795)    │   calib   │ camera f / boresight / per-photo pose │      │  collision: 2.5D ground │
│ ins/ppk.txt (INS)     │──────────▶│ ground : 2.5D height + semantic grid  │─────▶│  visual: textured tiles │
│ *.pcmp / *.pcpp       │   mesh    │ structure: 3D Poisson mesh per 20 m   │      │  LIMO + A* + pure pursuit│
└───────────────────────┘   bake    │ texture: photo → LAS rgb → material   │      │  top / chase / D455 RGB-D│
                                    └──────────────────────────────────────┘      └─────────────────────────┘
```

## 1. How to run

```bash
# 0) (first time only) prepare robot URDFs
conda activate umasstwin
python scripts/prepare_robots.py

# 1) Digital twin build — processing env. Cached per stage, so a rerun only does the missing stages.
python scripts/build_twin.py                       # configs/blue_wall.json, whole area
python scripts/build_twin.py --region 85 170 115 200 --name crop   # quick test on a 30 m area only
python scripts/eval_vo.py                          # (optional) SfM from photos only → compare with INS
python scripts/build_collision.py                  # structure mesh collision (8 s). With crop, also --variant crop
python scripts/build_collision.py --mode voxel     # structure voxel collision (47 s, recommended default. Section 6)

# 2) Navigation — Isaac Gym env
conda activate umasstwin
python scripts/run_nav.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/nav/run_000 --headless --collision voxel                  # voxel collision (recommended default)
python scripts/run_nav.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/nav/run_000_nocol --headless --collision none             # OFF (for mesh collision use --collision mesh or omit it)
```

- Waypoint coordinates are in local metres. Read them from `assets/twin/blue_wall/map.png` (green = drivable, 10 m grid).
  Use `--utm` to give them in UTM. The first point is the start, and points that are not drivable or touch an obstacle are moved
  to a nearby cell with 1 m of margin (the moved distance is printed).
- `--plan-only`: only makes `plan.png`, without the sim. Without `--headless` a viewer window opens.
- `--collision {mesh,voxel,none}`: 3D structure collision shape (section 6). `none` leaves only the visual (this is also the state when the corresponding npz file is missing).
- `--robot {warthog,limo,jackal,husky}` (default `warthog`, section 5d), `--speed`, `--cam-pitch-deg` (D455 downward angle), `--variant crop`.
- To use another session, create `configs/<name>.json` and pass it with `--config` (`session_dir`, `las`, `bbox_utm`, `origin`).

`run_nav.py` output (`--out` folder):

| File | Content |
|---|---|
| `combined.mp4` | top view + third person + D455 RGB (depth inset) in one frame |
| `topview.mp4`, `chase.mp4` | top view (planned path and driven trajectory overlay), third-person follow |
| `d455_rgb.mp4`, `d455_depth.mp4`, `depth/*.png` | 1280×720, 30 fps, HFOV 87°. depth is uint16 mm (0 = out of range, 0.52–20 m) |
| `trajectory.csv` | per-frame robot pose, camera pose, UTM, commanded and measured velocity |
| `plan.png`, `result.json`, `route.npy` | planned path map, arrival time and distance per waypoint, camera intrinsics |

## 2. Dataset structure (`~/mydata/worldmodel/rss2027`)

Data collected while walking with an Inertial Labs **RESEPI** (A84717) mobile mapping payload.

| Sensor | Content |
|---|---|
| LiDAR | Hesai PandarXT32 (32 lasers, 10 Hz). Spin axis is 45° forward-down in the body frame |
| Camera | Sony a5100, 16 mm, 6000×4000. 45° forward-down. gphoto **software trigger**, about 2.5 s interval |
| INS | NovAtel OEM7720 GNSS + IMU 200 Hz. PPK (tightly coupled) with the HAMP CORS base station |

| Path | Content | Used |
|---|---|---|
| `RESEPI-…-03-07-18-07-32/` | **Main session**. 33 min, 2.1 km walk (mean 1.06 m/s), 110 × 147 m | ✔ |
| `  ins/ppk.txt` | post-processed trajectory 125 Hz: GPSTime, lon, lat, ellipsoidal height, heading, pitch, roll | ✔ |
| `  ins/SBET.OUT`, `*-imu.imr` | 200 Hz binary of the same trajectory, raw IMU | — |
| `  camera/<serial>_<t0>_<t1>.jpg` | 796 photos. `t/1e6 − 1e9` = LAS `gps_time` (exposure start / end, µs) | ✔ |
| `  processed_ppk.pcmp / .pcpp` | LiDAR / camera lever arm, boresight, lens `FOV=24.55`, CRS | ✔ (initial values) |
| `  data/` | 57 raw logs, base station RINEX | — |
| `lidar-003.scan` (8 GB) | raw scan of the main session (PCMaster-only format) | — |
| `RESEPI-…-03-04-*` (2) | short test recordings, no post-processed trajectory | — |
| `../bluewall/blue_wall.las` (16 GB) | **The main session georeferenced with PCMasterProGL**. 443 M pts, UTM 18N + ellipsoidal height. `gps_time`, `point_source_id` (= laser number 0–31), intensity, RGB (only 37% coloured) | ✔ |

Facts confirmed:

- The LAS `gps_time` and the trajectory time share the same axis, so **the sensor position that fired each point** can be recovered.
- The LAS is **shifted by a constant (−3.456, −4.822, −0.088) m** relative to the trajectory (cause not confirmed, open question).
  With this value the LiDAR ring residual goes from 3.3° → 0.008° (3.1 below).
- Intensity is concentrated at 1 / 13 / 63 (1% / median / 99%) out of 0–255, and the per-laser median is nearly uniform at 11–15.
- **The correlation coefficient between 905 nm intensity and photo brightness is −0.07.** Colour or brightness cannot be made directly from intensity.
- Most of the ground is **regular paving tiles**. Feature matching between photos produces false correspondences shifted by one tile.

## 3. Pipeline principles

### 3.1 LAS ↔ trajectory registration (`umass_lidar/resepi.py`)

One laser of a spinning LiDAR traces a cone around the spin axis. Using the condition that (point direction · spin axis) in the body frame
must be constant per laser, the world offset between LAS and trajectory, the spin axis and the laser elevations are solved together.
This check also verifies the body attitude convention (x right, y forward, z up; `Rz(−heading)·Rx(pitch)·Ry(roll)`).
Result: spin axis = (0, 0.709, −0.705), laser elevation 12.9° … −14.8° (matches the XT32 spec).

### 3.2 Camera calibration (`umass_lidar/camera.py`)

```
Xc = dR_i · R(rvec) · R_CB0 · R_wb(t_i)ᵀ · (Xw − (p_b(t_i) + R_wb · lever)) + dt_i
u  = f · x/z · (1 + k1 r² + k2 r⁴) + cx
```

1. **Mount orientation** `R_CB0`: search 72 axis combinations by the correlation "photo brightness vs point colour assigned by PCMaster" (correct one 0.45, the rest ≤ 0.09).
2. **Photo ↔ LiDAR map registration**: block-match an image of the LiDAR points rendered from the photo viewpoint against the photo
   to make 2D–3D correspondences. Blocks whose peak is not unique because of the repeating tiles are discarded.
3. **Global optimization**: solve only focal and boresight. The principal point is indistinguishable from the boresight and was fixed;
   distortion and time offset did not improve when solved and were rejected (reverted automatically).
4. **Per-photo pose correction**: a 6-DoF correction per photo with the INS pose as prior (0.3°, 3 cm). This is because with a software trigger
   the exposure time is not exact and per-photo error remains. Corrections exceeding 4 times the prior are discarded.

| Stage | Median reprojection error (at 6000 px) |
|---|---|
| Initial values (`.pcpp` lever arm, f = 16 mm) | 134 px |
| Global optimization (f = 4049 px, boresight pitch −1.83°) | 24 px |
| Per-photo correction (applied to 389 photos, p95 0.33° / 2.8 cm) | 17 px |

17 px is about 1.7 cm at 4 m distance. The registration itself is block matching at 1/16 resolution, so this value includes measurement noise.
Report: `outputs/calib/blue_wall/report.json`.

### 3.3 VO / SfM comparison (`scripts/eval_vo.py`)

COLMAP sequential SfM was run from photos only, without INS or LiDAR, and compared with the INS trajectory.

| Item | Result |
|---|---|
| Registered photos | 488 of 795 (61%) |
| Number of models | split into 41 (largest 26 photos, median 12) |
| 32 good models (377 photos) | median ATE 2.1 cm after Sim(3) alignment, relative rotation error 0.5° |
| 6 wrong models (99 photos) | ATE 1 m or more (wrongly joined because of tile repetition) |
| Focal estimated by SfM | 4090 px (within 1% of the 4049 px of 3.2 and the nominal 4085 px) |

Interpretation: because of still photos at 2.5 s intervals (about 2.7 m apart, camera looking down) and repeating tiles, **photos alone cannot
join the whole trajectory into one**. Within short segments they agree with the INS at cm level, which cross-checks that the INS pose is
reliable. The twin uses "INS pose + LiDAR-map-based photo correction" (3.2). Raw-IMU-based VIO was not done
(with 0.4 Hz photos it equals IMU dead reckoning). Figure: `outputs/vo/blue_wall/sfm_vs_ins.png`.

### 3.4 Mesh (`las_terrain.py`, `structure.py`)

- **Ground (2.5D, 0.25 m grid)**: a z occupancy column is built per xy cell to separate ground / obstacle / roof / unobserved.
  The ground is "the lowest voxel holding at least 8% of the densest voxel in the column" (removes reflection ghosts on wet pavement).
  Used for collision and planning. Obstacle cells are low stubs raised only up to 0.6 m above ground.
- **Structure (3D)**: points at least 0.25 m above ground are reduced to 5 cm voxels, PCA normals are flipped toward
  **the sensor that fired the point**, and a surface is built per 20 m tile with screened Poisson.
  Faces more than 12 cm away from points are deleted. Visual only, and exported double-sided because Isaac Gym does not draw back faces.

### 3.5 Texture (`texture.py`)

For each texel (ground 1 cm, structure 3 cm) the world position and normal are computed and the colour is decided in this order.

| Rank | Source | Method |
|---|---|---|
| 1 | **photo** | project every photo that sees the texel within 12 m. Photos occluded according to the LiDAR depth map are excluded. Weight = (cos incidence angle / distance² × image-edge falloff)³ |
| 2 | **las_rgb** | the colour PCMaster assigned to the points. Used as is only where coloured points are dense |
| 3 | **las_rgb low frequency × intensity** | where coloured points are sparse: mean colour over 25 cm × intensity shading |
| 4 | **material** | where there is no colour information at all: material table colour × intensity shading |
| — | no_data | unscanned area: grey |

**Intensity adjustment**

1. Per-laser gain equalization (match each laser's median to the overall median)
2. Stretch the 1–99.5% range, then gamma 0.5 → 0–1
3. This value is not used as brightness (uncorrelated with photo brightness). It is used in only two ways.
   - **Material separation**: for each bin of (4 surface types × 12 intensity bins), collect the mean colour of texels that have a photo to build the **material table**.
     The surface types are flat ground / sloped ground / vertical structure / other structure. All colours come from photos of this site.
   - **Shading detail**: multiply by `sqrt(I_fine / I_smooth)` clamped to 0.7–1.35 (I_smooth is a mean over about 12–15 cm).

**Material / rendering**: the Isaac Gym camera uses only the base colour of a single texture. The capture day was overcast and shadows in the photos are weak, so
the lighting of the photos is baked into the texture as is, and the sim shows it unlit. Isaac Gym's shading was measured to be
`tex × intensity × (ambient + max(0, n·l))`, so intensity is set to 0.04 and ambient to 25
(`sim.set_baked_lighting`). The sky is a dome mesh in the colour of an overcast day. The per-texel source is kept in `<tile>_source.png`.

### 3.6 Navigation (`nav.py`, `sim.py`, `scripts/run_nav.py`)

On the semantic grid, non-ground cells and slopes of 15° or more are blocked and inflated by the robot radius, then A* on a 0.5 m grid
(with extra cost near obstacles) connects the path between waypoints. Tracking is pure pursuit + skid-steer velocity control.
If the robot cannot move 0.2 m in 8 s it ends as `STUCK`, if it flips over as `FLIPPED`, and this is recorded in `result.json`.

## 4. Decisions made together

| Item | Decision | Reason |
|---|---|---|
| Mesh shape | 2.5D ground + 3D structure | keeps driving stability, while walls and trees in the ego view have their real shape |
| Colour of areas without photos | per-material colour × intensity shading | clearly an estimated colour, and predictable |
| VIO/VO | photo correction based on INS + independent SfM comparison | INS is cm-level. SfM is for cross-validation |
| D455 rendering | Isaac Gym camera + lighting-baked texture | physics, the three views and depth agree within one program |

## 5. Results

**Twin (`assets/twin/blue_wall/full`, 234 × 260 m)**

| Item | Value |
|---|---|
| Ground | 36 tiles, texel 1 cm (about 3900 × 4330 px per tile), collision mesh 1.94 M faces |
| Structure | 117 tiles of 20 m, at most 40 k faces per tile, texel 3 cm |
| Visual mesh total | 6.07 M faces (structures are double-sided, so about 2× when rendered), 1.5 GB |
| Build time | about 40 min in total (3 min for export only on a rerun after caching). Cache `data/blue_wall` 21 GB |

Texel colour source (by texel count):

| Source | Ground (inside the scanned area) | Structure |
|---|---|---|
| photo | 20.0% | 5.6% |
| las_rgb | 6.2% | 18.9% |
| las_rgb low frequency × intensity | 24.3% | — |
| material (material table × intensity) | 49.5% | 75.6% |

31.9% of the whole ground bbox is unscanned area (no_data). Photos are concentrated on the ground around the walking route, so
the plaza and lawn have photo colour, while trees and the upper parts of buildings are mostly material colour (greyish).

**Navigation example (`outputs/nav/run_000`)**

```bash
python scripts/run_nav.py --robot limo --waypoints 95 150  103 166  125 172  138 168 --out outputs/nav/run_000 --headless
```

| Item | Value |
|---|---|
| Robot | LIMO (4.5 kg), top speed 1.0 m/s |
| Result | all 3 waypoints reached (33.8 s / 95.3 s / 117.8 s) |
| Planned / driven distance | 85.7 m / 90.0 m |
| Final goal error | 0.40 m (arrival radius 0.4 m) |
| Recording | 3534 frames (30 fps). 57% valid D455 depth pixels (sky and beyond 20 m are 0) |

Fixed during development: the whole plaza being classified as obstacle because of ghost points below the ground caused by reflections on wet tiles
(the ground was changed to "the lowest voxel with at least 8% of the column's maximum density"), waypoints snapping right next to an obstacle so that
the LIMO climbed onto a 0.6 m stub and flipped (waypoints are moved to a cell with 1 m of margin), and the tracker re-acquiring an already
driven segment on a route that doubles back (per-segment tracking).

## 6. Open questions

When one is resolved, the item moves into the results of the sections above.

1. **Why the LAS is shifted by (−3.46, −4.82) m from the trajectory.** Needs checking whether it is a datum / base-station coordinate correction at PCMaster export
   or an alignment to other data (airborne LiDAR). It does not affect registration inside the twin, but the absolute UTM coordinates differ.
2. **Measured camera intrinsics.** There is no calibration file, so only the focal was estimated. The meaning of `FOV=24.55` in `.pcpp` is also unknown.
   With a checkerboard calibration, distortion could be included too.
3. **Photo trigger delay.** With a software trigger the filename time and the actual exposure time may differ. This was absorbed by the per-photo correction,
   but checking whether hardware trigger / event records exist would make it more accurate.
4. **Areas without photos.** Whether additional capture (especially with the camera closer to horizontal) is planned. The current camera points 45° down, so
   upper walls and distant places have no photos.
5. **Trees, railings, glass.** Thin branches fragment in the Poisson mesh, and LiDAR passes through glass, leaving holes in walls.
   Whether to handle trees as billboards / omit them, and whether to fill glass surfaces with planes.
6. **Snow and transient objects.** It was captured in March, so snow is in the texture and terrain. Whether to remove it or match the season.
7. **Structure collision.** Currently only the 0.6 m stubs collide. A tall robot or a manipulator would need a 3D collision mesh.
8. **D455 mount position and angle on the real LIMO.** Currently assumed (0.10, 0, 0.13) m from base, pitch 0°.
9. **A more realistic ego view.** Isaac Gym has no PBR, shadows or reflections. Whether to attach an external renderer using the recorded camera poses
   (e.g. Gaussian splatting initialized with LiDAR depth) or move to Isaac Sim.
10. **D455 sensor model.** Depth is ideal (no stereo noise, holes or IR pattern). Whether a noise model is needed.
11. **Remaining data.** The `03-04` sessions and the raw `.scan` were not used. Whether they are needed.

## 7. Known limitations

- Photo registration accuracy (about 1–2 cm) is similar to the ground texel (1 cm), so there is slight blur at the borders of overlapping photos.
- Per-photo exposure differences were not corrected (auto ISO). Brightness steps may be visible at photo borders.
- The ground mesh has 0.25 m spacing, so curb and stair edges are rounded.
- The robot is rendered unlit and looks flat.
- The collision stubs (0.6 m) are also in the visual mesh, so the lower part of walls and planters looks like a thick blob in places.
- Tiles with very many points use a lot of memory during bake (up to about 30 GB with 6 workers).

---

# Auxiliary tools: cropping a place out of airborne LiDAR

Early code for data seen only from above, such as `~/Desktop/Airborne.las` (20 GB, 706 M pts, 4.0 × 3.8 km, about 92 pts/m²).
Separate from the twin pipeline.

```bash
conda activate umasstwin
# 1) full overview (1 m/px, with UTM grid) → read the UTM coordinates of the area of interest
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview --res 1.0
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview --mark 703437 4696145 176
# 2) crop (+ 2.5D terrain GLB)
python scripts/crop_las.py --las ~/Desktop/Airborne.las --center 703437 4696145 --size 200 \
    --out data/site1.npz --glb site1
```

- `overview.json`: convert pixels to UTM with `x = xmin + col·res`, `y = ymax − row·res`.
- The crop streams 20 M points at a time and compares only the bbox in int32 raw coordinates (about 30 s for the full 20 GB).
- `--glb`: 2.5D heightfield mesh (`umass_lidar/mesh.py`). `--mode dsm` (highest point per cell), `dtm` (ground only),
  `hybrid` (ground + buildings); without classification only `dsm` is possible.
- `~/Desktop/b1.e57` stores UTM coordinates as float32, quantized to 0.5 m, so it is not used.

# Environments

One conda env, `umasstwin` (Python 3.8), runs everything: data processing and the Isaac Gym simulation.
It is described by [environment.yml](environment.yml).

```bash
conda env create -f environment.yml
conda activate umasstwin
# Isaac Gym Preview 4 is an NVIDIA download (https://developer.nvidia.com/isaac-gym), not on PyPI
pip install --no-deps -e ~/Downloads/isaacgym/python
# Isaac Gym needs the env's libpython on the library path; store that in the env once
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib && conda activate umasstwin
```

- Python is 3.8 because Isaac Gym Preview 4 supports nothing newer. That pins numpy to 1.23 and a few packages
  (`lazrs`, `pycollada`, `tifffile`) to their last Python 3.8 releases; the versions are fixed in `environment.yml`.
- `--no-deps` skips PyTorch, which Isaac Gym lists as a requirement but this repo never imports (no `gymtorch`).
- After the `config vars` line, `conda activate umasstwin` is all a new shell needs; no `export LD_LIBRARY_PATH` any more.
- Needs Linux and an NVIDIA GPU. Blender (`scripts/blender_twin.py`) is a separate install and runs with its own Python.

**Checked in `umasstwin`** on a 120 m test site: `select_region.py` → `build_twin_site.py` (ortho ground) → `build_collision.py`
(mesh and voxel) → `prepare_robots.py` (with `--starling`) → `run_fleet.py` (headless, Warthog + Starling, both reached their goals).
**Not re-run there**: `build_twin.py` / `build_twin_aerial.py` (the RESEPI and Metashape pipelines) and `eval_vo.py`, which
were developed in the older envs below; `eval_vo.py` was written against pycolmap 4.2 and the Python 3.8 wheel is 3.12.

<details><summary>Older two-env setup (still works on the original machine)</summary>

| Purpose | conda env |
|---|---|
| Data processing | `umasslidar` (miniforge, py3.10): `source ~/miniforge3/etc/profile.d/conda.sh && conda activate umasslidar` |
| Simulation | `gleam` (anaconda3, py3.8): `source ~/anaconda3/etc/profile.d/conda.sh && conda activate gleam && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |

The two envs live in different conda installs, and a shell only sees the envs of the install whose `conda.sh` it sourced
last, which is what `EnvironmentNameNotFound` means.

</details>

Robot URDFs are built by `scripts/prepare_robots.py` without ROS (Warthog, Jackal and Husky clone the Clearpath repos,
LIMO uses `~/Documents/limo/agilex_open_class/limo/limo_description`). After xacro → URDF and DAE → OBJ conversion it
attaches `d455_link` and records wheel radius, track and camera link in `assets/robots/robots.json`.

# Repo structure

```
configs/blue_wall.json      session config (session_dir, las, bbox_utm, origin)
scripts/
  build_twin.py             session → digital twin (prep / calib / ground / bake / export)
  edit_drivable.py          hand-drawn allow / block polygons over the automatic drivable area
  run_teleop.py             keyboard teleoperation of one UGV or one UAV (Isaac Gym)
  run_fleet.py              main experiment: N UGVs + M UAVs, waypoint picking on the map, live trajectories (Isaac Gym)
  run_nav.py                UGV waypoint navigation + three-view recording (Isaac Gym)
  run_uav.py                UAV (Starling / quad) waypoint flight + RGB-D recording (Isaac Gym)
  run_team.py               UGV + UAV running together (same A* path) + RGB-D recording of both robots (Isaac Gym)
  build_collision.py        3D structure static collision: mesh mode (simplified visual tiles) / voxel mode (point cloud → closed voxel shell)
  eval_vo.py                SfM from photos only → comparison with INS
  prepare_robots.py         robot URDF preparation (--starling fetches the Starling 2 Max meshes)
  select_region.py          pick an area on the overview map and crop it to a LAS
  build_twin_site.py        cropped coloured point cloud LAS → digital twin (no photos or trajectory)
  build_twin_aerial.py      Metashape aerial photo project → digital twin (EE2023, configs/blue_wall_aerial.json)
  blender_twin.py           open a twin in Blender to tune texture / D455 (docs/blender_tuning.md)
  las_overview.py, crop_las.py   auxiliary tools (airborne LiDAR)
umass_lidar/
  resepi.py                 session loader, UTM conversion, LAS shift estimation, tile/voxel cache
  camera.py                 camera model, photo↔LiDAR registration, calibration
  las_terrain.py            LAS → 2.5D ground / obstacle / roof
  structure.py              3D structure mesh (Poisson)
  texture.py                photo projection, intensity model, material table, GLB export
  metashape.py              Metashape chunk loader (pose, depth map → cloud, aerial photo bank)
  nav.py                    traversability, A*, pure pursuit
  waypoint_ui.py            waypoint picking window on the top-view map (used by run_fleet.py)
  uav.py                    quadrotor URDF, rotor dynamics, geometric controller, 3D flight path
  sim.py, sim_assets.py     Isaac Gym helpers (twin loading, lighting, robot, camera)
  viz.py, mesh.py, io.py    map drawing, 2.5D mesh, I/O for the auxiliary tools
assets/twin/<name>/         ground.glb (collision) + <variant>/ tile GLB / PNG / twin_visual.urdf (generated, not in git)
data/<name>/                cache: tiles, voxel, depth map, camera.json (not in git)
outputs/                    calib / vo / nav results (not in git)
```
