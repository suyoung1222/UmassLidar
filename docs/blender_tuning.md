# Blender로 보면서 texture / 센서 fine-tuning 하기

Digital twin(`assets/twin/<name>/<variant>`)을 Blender에서 열어, Isaac Gym을 매번 돌리지 않고
texture 색과 D455 카메라 설정을 눈으로 확인하면서 조정하는 방법이다.
대상 twin은 세 가지이고 방법은 같다.

| twin | config | build script | 예시 run |
|---|---|---|---|
| LiDAR + 지상 사진 (`blue_wall`) | `configs/blue_wall.json` | `scripts/build_twin.py` | `outputs/nav/run_000` |
| 항공 사진 (`blue_wall_aerial`) | `configs/blue_wall_aerial.json` | `scripts/build_twin_aerial.py` | `outputs/nav/aerial_001` |
| 위에서 본 색 점군 (`library`) | `configs/library.json` | `scripts/build_twin_site.py` | `outputs/nav/library_000` |

조정 loop는 세 단계로 속도가 다르다. 가능한 한 위쪽 loop에서 끝낸다.

| loop | 무엇을 바꾸나 | 한 번 도는 시간 | 반영 방법 |
|---|---|---|---|
| A. Blender 안에서 | 전체 밝기·채도·gamma, 조명, 카메라 FOV·장착 위치·pitch | 즉시 | 값만 config / 코드에 옮겨 적는다 |
| B. export만 다시 | `grade`, seam feather, no-data 색, 재질표 | 항공 twin 약 15 s, LiDAR twin은 더 걸림(미측정) | build script 재실행 → Blender에서 image reload |
| C. bake 다시 | 사진 선택·가림 판정·blend 가중치·texel 크기·mesh | tile당 수십 초 (`--region`으로 일부만) | `--redo bake` → Blender script 재실행 |

## 1. 열기

Blender 5.3(`/snap/bin/blender`)에서 확인했다. conda env와 무관하다.

```bash
# GUI: twin 전체 + run_000의 D455 궤적
blender --python scripts/blender_twin.py -- --twin assets/twin/blue_wall/full --run outputs/nav/run_000

# 일부 영역만 (local metre, map.png에서 읽은 좌표)
blender --python scripts/blender_twin.py -- --twin assets/twin/blue_wall_aerial/full \
    --run outputs/nav/aerial_001 --region 70 130 170 200

# 창 없이 D455 시점 still + depth 렌더 (trajectory frame 0, 300, 600, ...)
blender -b --python scripts/blender_twin.py -- --twin assets/twin/blue_wall/full \
    --run outputs/nav/run_000 --render outputs/blender/run_000 --frames 0 3000 300 --depth
```

| option | 의미 |
|---|---|
| `--twin` | twin variant 폴더 (필수) |
| `--run` | `trajectory.csv` + `result.json`이 있는 nav 결과 |
| `--at X Y YAW` | 주행 기록 없이 D455를 지면 위 `--cam-height`(기본 0.28 m, LIMO 실측)에 놓는다. 여러 번 쓰면 frame 0, 1, ... |
| `--region X0 Y0 X1 Y1` | 겹치는 tile만 불러온다. `full` variant에서만 정확하다 |
| `--no-structure` | 바닥 tile만 |
| `--lit 0..1` | 0 = baked 색 그대로(Isaac Gym과 같은 모습), 1 = Sun + 하늘로 다시 조명 |
| `--engine eevee\|cycles` | 기본 eevee |
| `--packed` | PNG 파일 대신 GLB 안의 texture를 쓴다 (reload 불가) |
| `--render DIR --frames S E STEP` | still 렌더 후 종료. `--depth`면 `<frame>_depth.exr`도 저장 |
| `--save X.blend` | scene 저장 (mesh가 들어가 수백 MB) |

불러온 scene의 구성:

- collection `ground`(g_a_b), `structure`(s_ix_iy). 좌표는 simulator와 같은 local metre (x 동, y 북, z 위).
  UTM = local + `twin.json`의 `origin`.
- 모든 tile material이 node group **`TwinShade`** 하나를 공유한다 → knob 하나로 전체가 바뀐다.
- texture는 `<tile>.png` 파일을 link한다 → 파일이 바뀌면 reload만 하면 된다.
- empty `d455_link`(robot의 카메라 link pose, frame마다 keyframe) 아래에 카메라 `D455`.
- 색 관리는 `Standard`로 고정했다. AgX / Filmic으로 바꾸면 baked 색이 달라 보이니 바꾸지 않는다.

화면 조작: `Numpad 0` = D455 시점, timeline을 끌면 주행 경로를 따라 이동(frame = `trajectory.csv`의 frame, 30 fps),
viewport shading은 Material Preview 또는 Rendered, `F12` = 현재 frame 렌더.

Isaac Gym과의 일치 확인(aerial_001 frame 2850): RGB는 pose가 pixel 단위로 겹치고,
depth는 median 0.5 mm / p90 5 mm 차이. 따라서 Blender에서 본 시점은 Isaac Gym D455와 같다고 보면 된다.

## 2. Texture

### 2.1 즉시 조정: `TwinShade` knob

Shader editor에서 아무 tile material의 `TwinShade` node를 선택하고 `Tab`으로 들어가면 Value node 5개가 있다.
Python console(Scripting tab)에서 바꿔도 된다.

```python
K = bpy.data.node_groups["TwinShade"].nodes
K["Gain"].outputs[0].default_value = 1.3        # 밝기 배율 (linear RGB)
K["Saturation"].outputs[0].default_value = 1.15 # 1 = 원본, 0 = 흑백
K["Gamma"].outputs[0].default_value = 0.9       # <1 밝은 쪽으로, >1 어두운 쪽으로
K["Lit"].outputs[0].default_value = 0.0         # 0 baked / 1 Sun+sky relight
K["ShowSource"].outputs[0].default_value = 1.0  # 1 = texel 출처 mask 표시
```

마음에 드는 값을 config JSON에 적으면 export 단계가 같은 식(linear RGB에서 gain → saturation → gamma,
`texture.apply_grade`)으로 PNG / GLB에 굽는다. Isaac Gym에도 그대로 반영된다.

```json
"grade": {"gain": 1.3, "saturation": 1.15, "gamma": 0.9}
```

```bash
python scripts/build_twin_aerial.py          # 항공 twin: bake cache가 있으면 export만, 약 15 s
python scripts/build_twin.py                 # LiDAR twin: texture가 커서 더 걸린다 (미측정)
```

굽고 난 뒤 Blender에서는 knob을 1로 되돌리고(이중 적용 방지) image를 다시 읽는다.

```python
for im in bpy.data.images: im.reload()
```

`grade`는 twin 전체에 한 번 곱하는 값이다. 사진 구역과 재질 fallback 구역의 **상대** 밝기는 바꾸지 못한다(2.3 참고).

### 2.2 출처 mask 읽기 (`ShowSource = 1`)

| 색 | 출처 | 의미 |
|---|---|---|
| 초록 | photo | 사진이 직접 투영된 texel. 품질 기준이 되는 영역 |
| 파랑 | las_rgb (항공 twin에서는 point_rgb) | 사진에서는 가려졌고 주변 point 색으로 채움 |
| 연파랑 | las_rgb_lowfreq × intensity | 색 있는 point가 드물어 저주파 색 + intensity 무늬 (LiDAR twin만) |
| 주황 | material | 측정된 색 없음. 재질표(표면 종류 × intensity bin의 평균 사진색) |
| 회색 | no_data | scan 밖 |

보는 요령: 색이 어색한 곳에서 mask를 켜서 어느 출처인지 먼저 확인한다.
초록인데 어긋나 있으면 calibration / 가림 판정 문제(2.3의 visibility, 3.3),
초록↔주황 경계가 눈에 띄면 feather나 재질표 문제, 주황이 넓으면 사진이 닿지 않은 구역이다.

### 2.3 다시 구워야 하는 parameter

export만 다시 (loop B) — `umass_lidar/texture.py`

| 위치 | 기본값 | 효과 |
|---|---|---|
| `finish_texture(feather_px=6)` | 6 texel | 측정 texel ↔ 재질색 경계를 섞는 폭. 경계선이 보이면 키운다 |
| `finish_texture(nodata_rgb=(96,96,96))` | 회색 | scan 밖 색 |
| `MaterialTable.table(min_count=2000)` | 2000 | bin 하나를 믿는 최소 texel 수. 재질색이 튀면 키운다 |
| `N_BINS = 12` | 12 | intensity bin 수 (바꾸면 bake부터 다시) |
| `_detail()` clip `0.7 ~ 1.35` | | 재질색 위에 얹는 intensity 무늬의 대비 |
| `write_sky()` `base = 228 - 26*...` | | 하늘 dome 밝기 (Isaac Gym 전용, Blender는 World 색) |

bake 다시 (loop C) — LiDAR twin, `texture.py`

| 위치 | 기본값 | 효과 |
|---|---|---|
| `--tex-res` (build_twin.py) | 0.01 m | 바닥 texel. 사진 해상도 한계보다 작게 해도 선명해지지 않는다 |
| `--struct-texel` | 0.03 m | 구조물 texel |
| `PhotoBank(reduce=4)` | 1/4 | 사진 축소율. 2로 하면 선명해지지만 RAM 4배 |
| `PhotoBank(max_dist=12.0)` | 12 m | 이보다 먼 표면에는 사진을 쓰지 않는다. 늘리면 사진 영역이 넓어지고 흐려진다 |
| `sample()` `abs(z - dm) < 0.04*z + 0.06` | | 가림 판정 허용 오차. 줄이면 ghost가 줄고 구멍이 늘어난다 |
| `sample()` `cos > 0.15` | 약 81° | 이보다 비스듬히 본 사진은 버린다 |
| `sample()` fade `0.08 * width` | 8% | 사진 가장자리 감쇠 폭 (vignetting / 왜곡 잔차 회피) |
| `sample()` weight `cos * fade / z²` | | 사진 한 장의 가중치 |
| `project_photos(power=3.0)` | 3 | 가중치 지수. 크면 best 한 장에 가까워져 선명 + seam, 작으면 부드럽고 흐림 |
| `IntensityModel.fit(gamma=0.5)`, percentile `[1, 99.5]` | | intensity → [0,1] 변환. 재질 bin과 무늬에만 쓰인다 |
| ground `abs(dz) < 0.15` | 0.15 m | 바닥 texel에 색을 주는 point의 높이 허용 범위 |
| ground `dense` / `s_low = 0.25/tex_res` | | las_rgb를 선명하게 쓸지, 저주파로만 쓸지 가르는 밀도 기준 |
| structure `distance_upper_bound=0.12` / `0.2` | | 구조물 texel이 point 색 / intensity를 찾는 반경 |

bake 다시 — 항공 twin (`scripts/build_twin_aerial.py`, `umass_lidar/metashape.py`)

| 위치 | 기본값 | 효과 |
|---|---|---|
| `--tex-res` | 0.025 m | 사진 GSD가 약 3 cm라 더 줄여도 이득이 없다 |
| `--struct-texel` | 0.05 m | |
| `--voxel` | 0.25 m | 구조물 point voxel. 작으면 mesh가 조각나고(0.12에서 확인), 크면 뭉개진다 |
| `--min-h` | 0.4 m | 이 높이 이상만 3D 구조물로 mesh |
| `--max-faces` | 40000 | 20 m tile당 face 수 |
| `AerialBank(per_sector=5, sectors=8)` | 방위 8방향 × 5장 | tile마다 쓰는 사진 수. 늘리면 벽면 coverage↑, 시간·RAM↑ |
| `AerialBank.sample` `tol = 0.5 + 0.006*z` | | depth map 가림 판정 허용 오차 (depth map이 1/8 해상도라 넉넉하다) |
| `bake_ground` `support > 2.0`, `tol 0.4` | | point 색 fallback 기준 |

일부만 빠르게 굽기:

```bash
# LiDAR twin: 60 x 60 m만 crop variant로
python scripts/build_twin.py --region 80 130 140 190 --name crop --redo bake
blender --python scripts/blender_twin.py -- --twin assets/twin/blue_wall/crop --run outputs/nav/run_000

# 항공 twin
python scripts/build_twin_aerial.py --region 80 130 140 190 --name crop --redo bake
```

mesh나 UV가 바뀌는 변경(`--voxel`, texel 크기 등)은 image reload로는 안 되고 Blender script를 다시 실행해야 한다.
같은 pose에서 전후를 비교하려면 `--render`로 같은 `--frames`를 뽑아 나란히 놓는다.

### 2.4 조명

- `Lit = 0`: texture에 들어 있는 조명(LiDAR twin은 흐린 날, 항공 twin은 맑은 날 그림자 포함) 그대로.
  Isaac Gym의 `sim.set_baked_lighting(directional=0.04)`과 같은 모습이다. `directional`을 키우면 Isaac Gym에서
  위쪽 면만 밝아진다(robot 모델 입체감용).
- `Lit = 1`: `Sun` object(강도·방향)와 World 색으로 다시 조명한다. baked 그림자 위에 새 그림자가 겹치므로
  검토용이다. Isaac Gym에는 이 relight를 옮길 방법이 없다.

## 3. 센서

### 3.1 D455 (simulator 카메라)

Blender에서 `D455` 카메라를 고르고 값을 바꾸면 viewport(`Numpad 0`)에 바로 보인다. 확정한 값은 아래 위치에 옮긴다.

| Blender에서 | 의미 | 옮길 곳 |
|---|---|---|
| Camera > Lens > Field of View (sensor fit Horizontal) | HFOV, 기본 87° | `umass_lidar/sim.py` `D455["hfov"]` |
| Output > Resolution | 1280 × 720 | `D455["width"]`, `D455["height"]` |
| `D455` local Location | `d455_link` 기준 offset (x 앞, y 왼쪽, z 위). 기본 (0.015, 0, 0) | `sim.py` `RGBDCamera` `lt.p` |
| `d455_link` 높이·전방 위치 | robot `base_link` 기준 장착 위치. LIMO 기본 (0.10, 0, 0.13) | `scripts/prepare_robots.py` `cam_xyz` → script 재실행 |
| `D455`를 link의 Y축으로 회전 | pitch (양수 = 아래) | `run_nav.py --cam-pitch-deg` |
| Clip Start / End | 0.05 / 1500 m | `sim.py` `cp.near_plane`, `cp.far_plane` |

주의: `d455_link`의 keyframe은 기록된 주행의 pose다. 장착 위치를 시험할 때는 `d455_link`가 아니라 자식인
`D455`의 local transform을 움직인다(link를 움직이면 다음 frame에서 keyframe 값으로 돌아간다).

D455 실물 값과 맞출 때의 기준: RGB HFOV 약 90°(1280×800), depth HFOV 약 87°, depth 최소 거리 약 0.52 m(1280 폭).
현재 simulator는 RGB와 depth를 같은 pinhole(87°)로 뽑고 왜곡이 없다. 실제 LIMO에 단 D455의 장착 pose는 아직 모른다
(README Open questions).

### 3.2 Depth

`--depth`로 저장한 EXR은 R channel = planar z depth(m), A = 0이면 빈 곳이다. Isaac Gym의 `depth/*.png`(uint16 mm)와
같은 정의다. 유효 범위와 noise는 렌더 후처리이므로 Blender 없이 numpy로 조정한다.

```python
import numpy as np, OpenEXR, Imath
f = OpenEXR.InputFile("outputs/blender/run_000/000600_depth.exr")
dw = f.header()["dataWindow"]; w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
ch = lambda c: np.frombuffer(f.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), np.float32).reshape(h, w)
z, hit = ch("R"), ch("A") > 0.5
ok = hit & (z >= 0.52) & (z <= 20.0)                 # sim.py D455["depth_min"], ["depth_max"]
d16 = np.where(ok, z * 1000, 0).astype(np.uint16)    # run_nav.py depth PNG와 같은 형식
```

| 위치 | 기본값 | 의미 |
|---|---|---|
| `sim.py` `D455["depth_min"]` / `["depth_max"]` | 0.52 / 20.0 m | 범위 밖은 0(invalid) |
| `run_nav.py` `depth_vis(max_m=10.0)` | 10 m | 영상의 depth colormap 범위 (표시만) |

현재 depth는 이상적인 값이다(stereo noise, 구멍, edge 번짐 없음). noise model을 넣는다면 `RGBDCamera.read()`가
들어갈 자리이고, 위 snippet으로 Blender depth에 먼저 시험해 볼 수 있다.

Isaac Gym은 visual tile 외에 충돌 mesh `ground.glb`(2.5D 바닥 + 낮은 장애물 기둥)도 회색으로 그린다.
Blender scene에는 충돌 mesh가 없으므로, 구조물 mesh가 비어 있는 곳에서는 Isaac Gym에만 회색 기둥이 보일 수 있다.

### 3.3 촬영 센서 (twin을 만드는 쪽)

texture가 mesh와 어긋나 보이면(초록 mask 영역에서 선이 이중으로 보이거나 벽 무늬가 밀림) 색이 아니라 calibration이다.

| 대상 | 저장 위치 | 값 | 다시 구하기 |
|---|---|---|---|
| Sony a5100 intrinsic / boresight / 시간 offset | `data/blue_wall/camera.json` (`f`, `cx`, `cy`, `k1`, `k2`, `rvec`, `lever`, `dt`) | f = 4048.5 px, k1 = k2 = 0 | `build_twin.py --redo calib` |
| 사진별 pose 보정 | `data/blue_wall/camera.deltas.npy` | | 위와 같음 |
| LAS ↔ trajectory shift | `data/blue_wall/las_shift.json` | (−3.456, −4.822, −0.088) m | `--redo prep` |
| LiDAR intensity 변환 | `twin.json`의 `intensity_model` (laser별 gain, stretch, gamma) | gamma 0.5 | `IntensityModel.fit` |
| 항공 사진 calibration / pose | Metashape `doc.xml` (고정값, 읽기만 한다) | f ≈ 4381 px | 해당 없음 |
| 항공 cloud → LiDAR frame shift | `data/blue_wall_aerial/align_shift.json` | (−3.330, −4.769, −28.518) m | `build_twin_aerial.py --redo align` |

`camera.json`을 손으로 고쳐 시험할 수도 있다(예: `dt`, `k1`). 고친 뒤에는 `--region ... --name crop --redo bake`로
작은 영역만 구워 Blender에서 어긋남이 줄었는지 본다. `data/<name>/depth_s8/`(사진별 LiDAR depth cache)는
calibration에 의존하므로 `camera.json`을 바꿨으면 지우고 다시 만든다.

## 4. 위에서 본 색 점군으로 만든 twin (`build_twin_site.py`)

`select_region.py`로 자른 LAS는 사진도 센서 궤적도 없어서 texture 출처가 점 색 하나다. 조정할 것이 적다.

| 고려사항 | 조정 방법 |
|---|---|
| 점 색에 촬영 당시 햇빛·그림자가 들어 있다 | 지울 수 없다. 전체 톤만 `TwinShade` → config `"grade"` (loop A/B) |
| 점 간격 약 0.1 m가 해상도 한계 | `--tex-res`를 0.05보다 줄여도 이득 없음. 가까운 바닥은 흐리다 |
| 점박이 / 뭉개짐 | `--smooth` (기본 0.07 m). 점박이면 키우고 흐리면 줄인다 (`--redo bake`) |
| 나무 밑·처마 밑 빈 바닥 | `--fill` (기본 0.6 m) 폭으로 주변 바닥색을 번지게 한다. 그래도 빈 곳은 재질 평균색(주황 mask) |
| 벽면은 점이 성기다 | `--voxel` (기본 0.12 m)로 mesh 조밀도 조정. 구멍이 많으면 키운다 |
| 높은 건물 | tile 높이에 맞춰 Poisson depth가 8~10으로 자동 조정된다 |

이 twin에는 2.3절의 사진 관련 parameter(가림 판정, blend 가중치, calibration, intensity model)가 전혀 쓰이지 않는다.
`ShowSource`의 연파랑은 여기서는 "주변 바닥색으로 메운 곳"을 뜻한다.

Future work (UAV-UGV world model 학습)를 위해 지금 구조에서 알아둘 점:

- 지상(D455)과 공중 시점이 **같은 texture**를 본다. 위에서 찍은 색이라 공중 시점은 원본에 가깝고, 지상 시점은 벽·가까운 바닥이 흐리다.
  두 시점의 품질 차이가 학습에 영향을 줄 수 있다. 지상 품질을 올리려면 EE2023 drone 사진 투영(처리된 Region만)이나 지상 재촬영이 필요하다.
- 조명과 그림자가 구워져 있어 시간대·날씨 변화를 만들 수 없다.
- 움직이는 물체가 없고, 주차된 차·사람은 촬영 당시 모습 그대로 mesh에 굳어 있다.
- 공중 카메라는 `run_nav.py`에 아직 없다(top-down 기록 카메라만 있다). Blender에서는 `d455_link`를 임의 높이로 올려 미리 볼 수 있다.
- 좌표는 config의 `origin` + local m로 UTM과 이어지므로 UAV 궤적(GPS)과 같은 frame에서 다룰 수 있다. 높이는 이 LAS의 표고 기준이다.

## 5. 알아둘 차이와 한계

- Isaac Gym은 가까운 바닥 texture가 Blender보다 흐리다(texture filtering 차이). 선명도 판단은 Isaac Gym 영상으로 마지막에 확인한다.
- Isaac Gym은 뒷면을 그리지 않아 구조물 GLB는 면을 양면으로 복제해 둔다. Blender에서는 face 수가 2배로 보인다.
- 바닥 tile은 충돌 mesh 위로 0.015 m 띄워져 있다(`twin_visual.urdf`와 같은 값).
- `--region`은 ground tile 범위를 6 × 6 분할로 가정한다(`--ground-split` 기본값). `crop` variant에는 쓰지 않는다.
- EXR depth는 half float로 저장되어 20 m에서 약 1 cm 단위다.
- 항공 twin: 구조물 tile `s_10_0`(영역 모서리)은 bake worker가 죽어 빠져 있다. 벽면은 사진 각도가 얕아 흐리고,
  바닥에는 촬영 당시 그림자가 구워져 있다.
- LiDAR twin 전체를 `--save`하면 .blend가 200 MB를 넘는다. `/home` 여유 공간이 적으니 필요할 때만 저장한다.
