# UmassLidar

UMass Amherst 캠퍼스 LiDAR 데이터로 Isaac Gym용 digital twin을 만들고, 그 안에서 wheeled robot의
navigation 실험과 주행 데이터 수집을 하는 repo.

- **Main pipeline**: 지상 수집 세션(LiDAR + 사진 + INS) → textured digital twin → LIMO waypoint navigation.
  아래 "RSS2027 Digital Twin" 절 전체가 이 내용이다.
- **보조 도구**: 항공 LiDAR(`Airborne.las`)에서 장소를 잘라 2.5D terrain을 만드는 초기 실험 코드. 맨 아래 절.


# Quick guideline
## 1. Select region.

```bash
conda activate umasslidar
# (최초 1회) 전체 overview 지도 생성
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview
# 지도에서 네모를 그려 그 영역을 LAS로 잘라낸다
python scripts/select_region.py --name library
```

- 조작: 왼쪽 drag = 네모(모서리를 끌어 수정), scroll = zoom, 오른쪽 drag = 이동, `Enter` = 잘라내기, `Esc` = 취소.
- 제목 줄에 크기 / 면적 / 예상 point 수가 나온다. 한계를 넘으면 빨간색이 되고 `Enter`가 거부되니 다시 그린다.
- 출력: `data/sites/<name>.las` (원본과 같은 형식·좌표계) + `<name>.json` (bbox, 면적, point 수). 20 GB 전체 scan에 약 30초.
- 기본 한계: 한 변 500 m, 25 ha (RTX 4090 24 GB 기준 추정). `--max-side`, `--max-area`로 바꾼다.
- 다른 LAS는 `--las`와 `--overview`, 출력 위치는 `--out`.
- 화면이 없으면(원격) 좌표로 직접: `python scripts/select_region.py --bbox XMIN YMIN XMAX YMAX --name b1`

## 2. Build twin from the region

```bash
conda activate umasslidar
python scripts/build_twin_site.py --las data/sites/library.las      # library 9.9 ha: 수 분
```

- 입력은 점의 위치와 색뿐이다(사진·센서 궤적 불필요). 첫 실행 때 `configs/library.json`이 만들어지고, 이후 모든 script는 이 config를 쓴다.
- 출력: `assets/twin/library/` (`ground.glb` 충돌 mesh, `map.png` waypoint용 지도, `full/` tile GLB·PNG·`twin.json`), cache는 `data/library/`.
- 단계별 cache: 다시 실행하면 export만 돈다(약 10 s). 다시 계산하려면 `--redo ground|cloud|bake`.
- 일부만 빠르게: `--region X0 Y0 X1 Y1` (local m, `map.png` 좌표) → `crop` variant로 따로 저장된다.
- 품질 option (바꾸면 `--redo bake`):
  `--tex-res 0.05` 바닥 texel(점 간격이 약 0.1 m라 더 줄여도 선명해지지 않음), `--smooth 0.07` 점 색 번짐 폭,
  `--fill 0.6` 나무 밑 등 빈 바닥을 주변 색으로 메우는 폭, `--voxel 0.12` 구조물 mesh 조밀도(작으면 조각남, 크면 뭉개짐),
  `--min-h 0.4` 이 높이 이상만 3D 구조물.
- `map.png`의 초록 = 주행 가능. 건물·나무·주차된 차는 장애물로 잡힌다.

## 3. Compare texture using Blender

```bash
# 주행 기록 없이: 지도 좌표 (x y yaw°)에 LIMO 카메라 높이로 D455를 놓는다. yaw 0 = 동, 90 = 북
blender --python scripts/blender_twin.py -- --twin assets/twin/library/full --at 170 150 90

# 창 없이 여러 시점을 still로 (frame 0, 1, ... = --at 순서)
blender -b --python scripts/blender_twin.py -- --twin assets/twin/library/full \
    --at 170 150 90 --at 150 135 0 --render outputs/blender/library --frames 0 2 1

# 주행 기록이 있으면 그 궤적을 따라간다
blender --python scripts/blender_twin.py -- --twin assets/twin/library/full --run outputs/nav/library_000
```

- `Numpad 0` = D455 시점, viewport shading은 Material Preview. 색 관리는 `Standard`로 둔다.
- 색 조정: Shader editor에서 `TwinShade` group 안의 `Gain` / `Saturation` / `Gamma` 값을 바꾸면 전체 tile에 즉시 반영된다.
  정한 값을 `configs/library.json`에 `"grade": {"gain": 1.2, "saturation": 1.1, "gamma": 0.95}`로 적고 2번 명령을 다시 실행하면
  texture에 구워진다. 그 뒤 Blender에서는 knob을 1로 되돌리고 `for im in bpy.data.images: im.reload()`.
- 출처 확인: `ShowSource`를 1로. 파랑 = 점 색, 연파랑 = 주변 색으로 메운 곳, 주황 = 색 정보 없음(재질 평균색).
- mesh / 해상도 option을 바꿨을 때는 `--region`으로 작게 구워 `--twin assets/twin/library/crop`을 같은 `--at`으로 렌더해 비교한다.
- 전체 설명과 parameter 표: [docs/blender_tuning.md](docs/blender_tuning.md)

## 4. Run navigation

```bash
conda activate gleam && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python scripts/run_nav.py --config configs/library.json --waypoints 150 140 190 130 215 100 240 60 \
    --out outputs/nav/library_000 --headless
```

- waypoint는 `assets/twin/library/map.png`에서 읽은 local m (첫 점이 출발점). `--plan-only`로 경로만 먼저 확인할 수 있다.



---
# Datasets
## 1. 2026 03 07 snowy bluewall
Sensors: Inertial Labs RESEPI + Hesai 32ch LiDAR + Sony a5100
Data (https://drive.google.com/drive/u/1/folders/1MhfiWRgno-qihtz96PdWmms9i4fMCdGZ):
2026-03-07-18-07-32 (24 GB)	33분 주행, 사진 796장	PPK 후처리까지 끝남. 메인 세션
2026-03-04-23-35-52 (1.5 GB)	20초, 사진 11장, lidar.scan 94 MB	raw만 있음 (ppk.txt 없음). 테스트 촬영임.

```bash
camera/<serial>_<t0>_<t1>.jpg   796장, 6000×4000, Sony ILCE-5100
                                t는 µs, t/1e6 − 1e9 = LAS gps_time. EXIF orientation=3 (180° 회전)
ins/ppk.txt                     PPK 궤적 125 Hz: Week GPSTime UTCTime Lon Lat H-Ell Heading Pitch Roll
ins/trajectories/ppk/*.txt      같은 궤적을 geoid만 바꿔 5벌 (WGS84 ellipsoid, GEOID18 등)
ins/SBET.OUT                    같은 궤적 (SBET binary)
ins/*-imu.imr, *-gnss.gps 외     raw IMU/GNSS와 Waypoint Inertial Explorer 중간 파일
data/hamp0660.*, igu*.sp3       PPK 베이스국 데이터 (CORS 'hamp', RINEX obs/nav, 정밀 궤도력)
data/RESEPI-*.data (57개, 15 GB) 장비 raw 로그
processed_ppk.pcmp              LiDAR lever arm, boresight, 장착각(pitch −45°), 레이저별 보정
processed_ppk.pcpp              카메라 lever arm (0, −0.08, −0.048), boresight, 장착 −45°, 사진 목록
log.txt                         수집 요약: GNSS 위성 수(21–25개가 70%), IMU 통계, range 필터 8–200 m
```

## 2. Airborne.las EE2023
Sensors: Inertial Labs RESEPI + Hesai 32ch LiDAR + DJI drone camera
Data (https://drive.google.com/drive/u/1/folders/1kBTbf2Wi2AtV4I8o0_BCfMtxHIy65xED):

```bash
Aerial Images/RegionXX/DJI_*.JPG   25개 region, 사진 12,514장
    대부분 Mavic 2 Pro (5472×3648, RTK 없음, 고도 약 122 m), 수직 촬영(nadir)
    Region17/18/19/23/24/25/26/WEB는 경사 촬영 (gimbal −65°)
    Region08_09_10_15_16만 Mavic 3E + PPK 파일 (.nav/.obs/Timestamp)
AlignedChunksViaPointClouds/
    *_SubregionSet_1.psx/.files   chunk 17(R29), 18(R22), 38(R36_37_38_44), 40/41 merged
    *_SubregionSet_2_*.psx/.files chunk 10(R26), 14(R25), 15(R24), 16(R23), 37–39 merged
      각 chunk: chunk.zip/doc.xml (정렬된 카메라 pose, calibration, chunk→ECEF 변환)
                0/depth_maps (EXR, 원본의 1/8 해상도), dense_cloud (proprietary .oc3)
                point_cloud (tie points), merged chunk에는 elevation (DEM)
    transects.shp, Set2_Chunks*.png (GCP 정렬 전후 스크린샷)
```


---

# RSS2027 Digital Twin (main pipeline)

RESEPI payload로 걸어서 수집한 세션 하나를 입력으로 받아, 사진을 입힌 3D 환경을 만들고
Isaac Gym에서 LIMO 1대를 waypoint로 주행시키며 **탑뷰 / 3인칭 / RealSense D455 RGB-D** 세 시점을 기록한다.

```
RESEPI session                         build_twin.py (env: umasslidar)                run_nav.py (env: gleam)
┌───────────────────────┐   prep    ┌──────────────────────────────────────┐      ┌─────────────────────────┐
│ LAS  (LiDAR, 443 M pt)│──────────▶│ LAS↔trajectory shift, tile/voxel cache│      │ Isaac Gym               │
│ camera/*.jpg (795)    │   calib   │ camera f / boresight / per-photo pose │      │  collision: 2.5D ground │
│ ins/ppk.txt (INS)     │──────────▶│ ground : 2.5D height + semantic grid  │─────▶│  visual: textured tiles │
│ *.pcmp / *.pcpp       │   mesh    │ structure: 3D Poisson mesh per 20 m   │      │  LIMO + A* + pure pursuit│
└───────────────────────┘   bake    │ texture: photo → LAS rgb → material   │      │  top / chase / D455 RGB-D│
                                    └──────────────────────────────────────┘      └─────────────────────────┘
```

## 1. 실행 방법

```bash
# 0) (최초 1회) robot URDF 준비
conda activate umasslidar
python scripts/prepare_robots.py

# 1) Digital twin build — 처리 env. 단계별로 cache되므로 다시 실행하면 빠진 단계만 돈다.
python scripts/build_twin.py                       # configs/blue_wall.json, 전체 영역
python scripts/build_twin.py --region 85 170 115 200 --name crop   # 30 m 구역만 빠르게 시험
python scripts/eval_vo.py                          # (선택) 사진만으로 SfM → INS와 비교

# 2) Navigation — Isaac Gym env
conda activate gleam
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python scripts/run_nav.py --waypoints 95 150  103 166  125 172  138 168 \
    --out outputs/nav/run_000 --headless
```

- Waypoint 좌표는 local metre다. `assets/twin/blue_wall/map.png`(초록 = 주행 가능, 격자 10 m)에서 읽는다.
  UTM으로 주려면 `--utm`. 첫 점이 출발점이고, 주행 불가이거나 장애물에 붙은 지점은 1 m 여유가 있는
  가까운 cell로 옮겨진다 (옮긴 거리를 출력).
- `--plan-only`: sim 없이 `plan.png`만 만든다. `--headless`를 빼면 viewer 창이 뜬다.
- `--robot {limo,jackal,husky}`, `--speed`, `--cam-pitch-deg`(D455 아래 방향 각도), `--variant crop`.
- 다른 세션을 쓰려면 `configs/<name>.json`을 만들고 `--config`로 넘긴다 (`session_dir`, `las`, `bbox_utm`, `origin`).

`run_nav.py` 출력 (`--out` 폴더):

| 파일 | 내용 |
|---|---|
| `combined.mp4` | 탑뷰 + 3인칭 + D455 RGB(depth inset)를 한 화면에 |
| `topview.mp4`, `chase.mp4` | 탑뷰(계획 경로·주행 궤적 overlay), 3인칭 follow |
| `d455_rgb.mp4`, `d455_depth.mp4`, `depth/*.png` | 1280×720, 30 fps, HFOV 87°. depth는 uint16 mm (0 = 범위 밖, 0.52–20 m) |
| `trajectory.csv` | frame별 robot pose, camera pose, UTM, 명령·측정 속도 |
| `plan.png`, `result.json`, `route.npy` | 계획 경로 지도, waypoint별 도달 시각·거리, camera intrinsic |

## 2. 데이터셋 구조 (`~/mydata/worldmodel/rss2027`)

Inertial Labs **RESEPI** (A84717) mobile mapping payload를 들고 걸으며 수집한 데이터.

| 센서 | 내용 |
|---|---|
| LiDAR | Hesai PandarXT32 (32 laser, 10 Hz). spin axis가 body 기준 앞-아래 45° |
| 카메라 | Sony a5100, 16 mm, 6000×4000. 앞-아래 45°. gphoto **software trigger**, 약 2.5 s 간격 |
| INS | NovAtel OEM7720 GNSS + IMU 200 Hz. HAMP CORS 기준국으로 PPK (tightly coupled) |

| 경로 | 내용 | 사용 |
|---|---|---|
| `RESEPI-…-03-07-18-07-32/` | **메인 세션**. 33분, 2.1 km 보행(평균 1.06 m/s), 110 × 147 m | ✔ |
| `  ins/ppk.txt` | 후처리 trajectory 125 Hz: GPSTime, lon, lat, 타원체고, heading, pitch, roll | ✔ |
| `  ins/SBET.OUT`, `*-imu.imr` | 같은 trajectory의 200 Hz binary, raw IMU | — |
| `  camera/<serial>_<t0>_<t1>.jpg` | 사진 796장. `t/1e6 − 1e9` = LAS `gps_time` (노출 시작·끝, µs) | ✔ |
| `  processed_ppk.pcmp / .pcpp` | LiDAR·카메라 lever arm, boresight, lens `FOV=24.55`, CRS | ✔ (초기값) |
| `  data/` | raw 로그 57개, 기준국 RINEX | — |
| `lidar-003.scan` (8 GB) | 메인 세션의 raw scan (PCMaster 전용 포맷) | — |
| `RESEPI-…-03-04-*` 2개 | 짧은 테스트 기록, 후처리 trajectory 없음 | — |
| `../bluewall/blue_wall.las` (16 GB) | **메인 세션을 PCMasterProGL로 georeference한 결과**. 443 M pts, UTM 18N + 타원체고. `gps_time`, `point_source_id`(= laser 번호 0–31), intensity, RGB(37%만 유색) | ✔ |

확인한 사실:

- LAS의 `gps_time`과 trajectory 시각이 같은 축이라, **포인트마다 그것을 쏜 센서 위치**를 복원할 수 있다.
- LAS는 trajectory 대비 **(−3.456, −4.822, −0.088) m 일정하게 이동**돼 있다 (원인 미확인, open question).
  이 값을 넣으면 LiDAR ring 잔차가 3.3° → 0.008°가 된다 (아래 3.1).
- Intensity는 0–255 중 1 / 13 / 63 (1% / 중앙값 / 99%)에 몰려 있고, laser별 중앙값은 11–15로 거의 균일하다.
- **905 nm intensity와 사진 밝기의 상관계수는 −0.07**이다. Intensity로 색이나 밝기를 직접 만들 수 없다.
- 바닥 대부분이 **규칙적인 포장 타일**이다. 사진 간 feature matching이 타일 한 칸씩 밀린 가짜 대응을 만든다.

## 3. Pipeline 원리

### 3.1 LAS ↔ trajectory 정합 (`umass_lidar/resepi.py`)

Spinning LiDAR의 laser 하나는 spin axis 둘레의 원뿔을 그린다. Body frame에서 (포인트 방향 · spin axis)가
laser별로 일정해야 한다는 조건으로, LAS와 trajectory 사이의 world offset·spin axis·laser elevation을 함께 푼다.
이 검사가 body 자세 convention(x 오른쪽, y 앞, z 위; `Rz(−heading)·Rx(pitch)·Ry(roll)`)의 검증도 겸한다.
결과: spin axis = (0, 0.709, −0.705), laser elevation 12.9° … −14.8° (XT32 사양과 일치).

### 3.2 카메라 calibration (`umass_lidar/camera.py`)

```
Xc = dR_i · R(rvec) · R_CB0 · R_wb(t_i)ᵀ · (Xw − (p_b(t_i) + R_wb · lever)) + dt_i
u  = f · x/z · (1 + k1 r² + k2 r⁴) + cx
```

1. **장착 방향** `R_CB0`: 축 조합 72가지를 "사진 밝기 vs PCMaster가 입힌 포인트 색" 상관으로 탐색 (정답 0.45, 나머지 ≤ 0.09).
2. **사진 ↔ LiDAR map 정합**: LiDAR 포인트를 사진 시점으로 렌더한 이미지와 사진을 block matching해서
   2D–3D 대응을 만든다. 반복 타일 때문에 peak가 유일하지 않은 block은 버린다.
3. **전역 최적화**: focal, boresight만 푼다. principal point는 boresight와 구분되지 않아 고정했고,
   distortion·시간 offset은 풀어도 나아지지 않아 기각됐다 (자동으로 되돌림).
4. **사진별 pose 보정**: INS pose를 prior(0.3°, 3 cm)로 두고 사진마다 6-DoF 보정. Software trigger라
   노출 시각이 정확하지 않아 사진별 오차가 남기 때문이다. Prior의 4배를 넘는 보정은 버린다.

| 단계 | 재투영 오차 중앙값 (6000 px 기준) |
|---|---|
| 초기값 (`.pcpp` lever arm, f = 16 mm) | 134 px |
| 전역 최적화 (f = 4049 px, boresight pitch −1.83°) | 24 px |
| 사진별 보정 (389장 적용, p95 0.33° / 2.8 cm) | 17 px |

17 px는 4 m 거리에서 약 1.7 cm다. 정합 자체가 1/16 해상도 block matching이라 이 값은 측정 잡음을 포함한다.
리포트: `outputs/calib/blue_wall/report.json`.

### 3.3 VO / SfM 비교 (`scripts/eval_vo.py`)

INS와 LiDAR 없이 사진만으로 COLMAP sequential SfM을 돌려 INS trajectory와 비교했다.

| 항목 | 결과 |
|---|---|
| 등록된 사진 | 795장 중 488장 (61%) |
| 모델 개수 | 41개로 분열 (최대 26장, 중앙값 12장) |
| 정상 모델 32개 (377장) | Sim(3) 정렬 후 ATE 중앙값 2.1 cm, 상대 회전 오차 0.5° |
| 잘못된 모델 6개 (99장) | ATE 1 m 이상 (타일 반복으로 잘못 붙음) |
| SfM이 추정한 focal | 4090 px (3.2의 4049 px, 공칭 4085 px와 1% 이내) |

해석: 2.5 s 간격 정지 사진(약 2.7 m 간격, 아래를 보는 카메라)과 반복 타일 때문에 **사진만으로는 전체
trajectory를 하나로 잇지 못한다**. 짧은 구간 안에서는 INS와 cm 수준으로 일치하므로 INS pose가 신뢰할 만하다는
교차 확인은 된다. Twin은 "INS pose + LiDAR map 기반 사진 보정"(3.2)을 쓴다. Raw IMU 기반 VIO는 하지 않았다
(0.4 Hz 사진으로는 IMU dead-reckoning과 같다). 그림: `outputs/vo/blue_wall/sfm_vs_ins.png`.

### 3.4 Mesh (`las_terrain.py`, `structure.py`)

- **바닥 (2.5D, 0.25 m grid)**: xy cell마다 z 점유 column을 만들어 ground / obstacle / roof / 미관측을 나눈다.
  지면은 "column에서 가장 조밀한 voxel의 8% 이상을 가진 가장 낮은 voxel"이다 (젖은 포장면의 반사 ghost 제거).
  충돌과 planning에 쓴다. 장애물 cell은 지면 위 0.6 m까지만 올린 낮은 stub이다.
- **구조물 (3D)**: 지면 위 0.25 m 이상의 포인트를 5 cm voxel로 줄이고, PCA normal을
  **그 포인트를 쏜 센서 방향**으로 뒤집은 뒤 20 m tile마다 screened Poisson으로 표면을 만든다.
  포인트에서 12 cm 넘게 떨어진 면은 지운다. 시각 전용이고, Isaac Gym이 뒷면을 그리지 않아 양면으로 내보낸다.

### 3.5 Texture (`texture.py`)

Texel(바닥 1 cm, 구조물 3 cm)마다 world 위치와 normal을 구하고 색을 다음 순서로 정한다.

| 순위 | 출처 | 방법 |
|---|---|---|
| 1 | **photo** | 12 m 이내에서 texel을 보는 모든 사진을 투영. LiDAR depth map과 비교해 가려진 사진은 제외. 가중치 = (cos 입사각 / 거리² × 화면 가장자리 감쇠)³ |
| 2 | **las_rgb** | PCMaster가 포인트에 입혀 둔 색. 유색 포인트가 조밀한 곳만 그대로 사용 |
| 3 | **las_rgb 저주파 × intensity** | 유색 포인트가 드문 곳: 25 cm 범위 평균색 × intensity 명암 |
| 4 | **material** | 색 정보가 전혀 없는 곳: 재질표 색 × intensity 명암 |
| — | no_data | 스캔되지 않은 영역: 회색 |

**Intensity 조절**

1. laser별 gain 균등화 (laser별 중앙값을 전체 중앙값에 맞춤)
2. 1–99.5% 구간 stretch 후 gamma 0.5 → 0–1
3. 이 값은 밝기로 쓰지 않는다 (사진 밝기와 무상관). 두 가지로만 쓴다.
   - **재질 구분**: (표면 종류 4가지 × intensity 12구간) 칸마다, 사진이 있는 texel의 평균색을 모아 **재질표**를 만든다.
     표면 종류는 평평한 바닥 / 경사 바닥 / 수직 구조물 / 그 외 구조물. 색은 전부 이 현장 사진에서 나온다.
   - **명암 detail**: `sqrt(I_fine / I_smooth)`를 0.7–1.35로 제한해 곱한다 (I_smooth는 약 12–15 cm 평균).

**재질 / 렌더링**: Isaac Gym 카메라는 texture 한 장의 기본색만 쓴다. 촬영일이 흐려 사진에 그림자가 약하므로
사진의 조명을 그대로 texture에 굽고, sim에서는 조명 없이 그대로 보이게 했다. Isaac Gym의 shading이
`tex × intensity × (ambient + max(0, n·l))`임을 측정으로 확인해서 intensity 0.04, ambient 25로 둔다
(`sim.set_baked_lighting`). 하늘은 흐린 날 색의 dome mesh다. Texel별 출처는 `<tile>_source.png`로 남는다.

### 3.6 Navigation (`nav.py`, `sim.py`, `scripts/run_nav.py`)

Semantic grid에서 ground가 아닌 cell과 경사 15° 이상을 막고 robot 반경만큼 부풀린 뒤, 0.5 m grid에서 A*
(장애물 근처 비용 가중)로 waypoint 사이 경로를 잇는다. 추종은 pure pursuit + skid-steer 속도 제어다.
8초 동안 0.2 m 이상 못 움직이면 `STUCK`, 뒤집히면 `FLIPPED`로 끝내고 `result.json`에 남긴다.

## 4. 함께 정한 것

| 항목 | 결정 | 이유 |
|---|---|---|
| Mesh 형상 | 바닥 2.5D + 구조물 3D | 주행 안정성은 유지하고 ego view의 벽·나무는 실제 형상으로 |
| 사진 없는 영역 색 | 재질별 색 × intensity 명암 | 추정색임이 분명하고 예측 가능 |
| VIO/VO | INS 기준 사진 보정 + 독립 SfM 비교 | INS가 cm급. SfM은 교차 검증용 |
| D455 렌더링 | Isaac Gym 카메라 + 조명 baked texture | 물리·세 시점·depth가 한 프로그램에서 일치 |

## 5. 결과

**Twin (`assets/twin/blue_wall/full`, 234 × 260 m)**

| 항목 | 값 |
|---|---|
| 바닥 | 36 tile, texel 1 cm (tile당 약 3900 × 4330 px), 충돌 mesh 194만 face |
| 구조물 | 20 m tile 117개, tile당 최대 4만 face, texel 3 cm |
| 시각 mesh 합계 | 607만 face (구조물은 양면이라 렌더 시 약 2배), 1.5 GB |
| Build 시간 | 전체 약 40분 (cache 후 재실행 시 export만 3분). cache `data/blue_wall` 21 GB |

Texel 색 출처 (texel 수 기준):

| 출처 | 바닥 (스캔된 영역 안) | 구조물 |
|---|---|---|
| photo | 20.0% | 5.6% |
| las_rgb | 6.2% | 18.9% |
| las_rgb 저주파 × intensity | 24.3% | — |
| material (재질표 × intensity) | 49.5% | 75.6% |

바닥 전체 bbox의 31.9%는 스캔되지 않은 영역(no_data)이다. 사진은 보행 경로 주변 바닥에 집중돼 있어
광장·잔디는 사진 색이고, 나무와 건물 윗부분은 대부분 재질색(회색 계열)이다.

**Navigation 예시 (`outputs/nav/run_000`)**

```bash
python scripts/run_nav.py --waypoints 95 150  103 166  125 172  138 168 --out outputs/nav/run_000 --headless
```

| 항목 | 값 |
|---|---|
| Robot | LIMO (4.5 kg), 최고 1.0 m/s |
| 결과 | waypoint 3개 모두 도달 (33.8 s / 95.3 s / 117.8 s) |
| 계획 / 주행 거리 | 85.7 m / 90.0 m |
| 최종 goal 오차 | 0.40 m (도달 판정 반경 0.4 m) |
| 기록 | 3534 frame (30 fps). D455 depth 유효 pixel 57% (하늘과 20 m 밖은 0) |

개발 중 고친 것: 젖은 타일의 반사로 지면 아래에 생긴 ghost 포인트 때문에 광장 전체가 장애물로 분류되던 문제
(지면을 "column 최대 밀도의 8% 이상인 가장 낮은 voxel"로 바꿈), waypoint가 장애물 바로 옆으로 snap되어
LIMO가 0.6 m stub에 올라타 뒤집히던 문제(waypoint를 1 m 여유가 있는 cell로 옮김), 되돌아가는 경로에서
추종기가 지나온 구간을 다시 잡던 문제(구간별 추종).

## 6. Open questions

해소되면 해당 항목을 위 절의 결과로 옮긴다.

1. **LAS가 trajectory에서 (−3.46, −4.82) m 이동된 이유.** PCMaster export 때의 datum/기준국 좌표 보정인지,
   다른 데이터(항공 LiDAR)에 맞춘 것인지 확인 필요. Twin 내부 정합에는 영향이 없지만 UTM 절대 좌표가 달라진다.
2. **카메라 intrinsic 실측값.** Calibration 파일이 없어 focal만 추정했다. `.pcpp`의 `FOV=24.55` 의미도 불명.
   Checkerboard calibration이 있으면 distortion까지 넣을 수 있다.
3. **사진 trigger 지연.** Software trigger라 파일명 시각과 실제 노출 시각이 다를 수 있다. 사진별 보정으로
   흡수했지만, hardware trigger/event 기록이 있는지 확인하면 더 정확해진다.
4. **사진이 없는 영역.** 추가 촬영(특히 카메라를 수평에 가깝게) 계획이 있는지. 현재 카메라는 45° 아래라
   벽 윗부분과 먼 곳은 사진이 없다.
5. **나무·난간·유리.** 가는 가지는 Poisson mesh가 조각나고, 유리는 LiDAR가 통과해 벽에 구멍이 난다.
   나무를 billboard/생략으로 처리할지, 유리면을 평면으로 메울지.
6. **눈·일시적 물체.** 3월 촬영이라 눈이 texture와 지형에 들어 있다. 제거하거나 계절을 맞출지.
7. **구조물 충돌.** 지금은 0.6 m stub만 충돌한다. 키 큰 robot이나 manipulator가 필요하면 3D 충돌 mesh가 필요.
8. **실제 LIMO의 D455 장착 위치·각도.** 현재 base에서 (0.10, 0, 0.13) m, pitch 0°로 가정.
9. **더 사실적인 ego view.** Isaac Gym은 PBR·그림자·반사가 없다. 기록한 camera pose로 외부 renderer
   (LiDAR depth로 초기화한 Gaussian splatting 등)를 붙이거나 Isaac Sim으로 옮길지.
10. **D455 센서 모델.** Depth가 이상적이다 (stereo noise, hole, IR pattern 없음). Noise model이 필요한지.
11. **나머지 데이터.** `03-04` 세션과 raw `.scan`은 쓰지 않았다. 필요 여부.

## 7. 알려진 한계

- 사진 정합 정밀도(약 1–2 cm)가 바닥 texel(1 cm)과 비슷해서, 겹치는 사진 경계에 약한 번짐이 있다.
- 사진별 노출 차이를 보정하지 않았다 (ISO 자동). 사진 경계에서 밝기 단차가 보일 수 있다.
- 바닥 mesh가 0.25 m 간격이라 연석·계단 모서리가 둥글다.
- Robot은 조명 없이 렌더되어 평평하게 보인다.
- 충돌 stub(0.6 m)이 시각 mesh에도 들어 있어, 벽·화단 아래쪽이 두툼한 덩어리로 보이는 곳이 있다.
- 포인트가 매우 많은 tile은 bake 중 메모리를 많이 쓴다 (worker 6개 기준 최대 약 30 GB).

---

# 보조 도구: 항공 LiDAR에서 장소 잘라내기

`~/Desktop/Airborne.las`(20 GB, 706 M pts, 4.0 × 3.8 km, 약 92 pts/m²)처럼 위에서만 본 데이터용 초기 코드.
Twin pipeline과는 별개다.

```bash
conda activate umasslidar
# 1) 전체 overview (1 m/px, UTM grid 표시) → 관심 영역의 UTM 좌표를 읽는다
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview --res 1.0
python scripts/las_overview.py --las ~/Desktop/Airborne.las --out data/overview --mark 703437 4696145 176
# 2) 잘라내기 (+ 2.5D terrain GLB)
python scripts/crop_las.py --las ~/Desktop/Airborne.las --center 703437 4696145 --size 200 \
    --out data/site1.npz --glb site1
```

- `overview.json`: `x = xmin + col·res`, `y = ymax − row·res`로 pixel을 UTM으로 바꾼다.
- Crop은 20 M 포인트씩 streaming하며 int32 raw 좌표로 bbox만 비교한다 (20 GB 전체 약 30초).
- `--glb`: 2.5D heightfield mesh (`umass_lidar/mesh.py`). `--mode dsm`(cell 최고점), `dtm`(지면만),
  `hybrid`(지면 + 건물)이고, classification이 없으면 `dsm`만 가능하다.
- `~/Desktop/b1.e57`은 UTM 좌표를 float32로 저장해 0.5 m 단위로 뭉개져 있어 쓰지 않는다.

# 환경

| 용도 | conda env | 비고 |
|---|---|---|
| 데이터 처리 (`build_twin.py`, `eval_vo.py`, `prepare_robots.py`, 보조 도구) | `umasslidar` (miniforge, py3.10) | `pip install "laspy[lazrs]" open3d trimesh pillow scipy opencv-python-headless xatlas pycolmap matplotlib xacro pycollada` |
| Simulation (`run_nav.py`) | `gleam` (anaconda3, py3.8, Isaac Gym Preview 4) | 실행 전 `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` |

Robot URDF는 `scripts/prepare_robots.py`가 ROS 없이 만든다 (Jackal·Husky는 Clearpath repo를 clone,
LIMO는 `~/Documents/limo/agilex_open_class/limo/limo_description` 사용). xacro → URDF, DAE → OBJ 변환 후
`d455_link`를 붙이고 `assets/robots/robots.json`에 바퀴 반경·track·카메라 link를 기록한다.

# Repo 구조

```
configs/blue_wall.json      세션 설정 (session_dir, las, bbox_utm, origin)
scripts/
  build_twin.py             세션 → digital twin (prep / calib / ground / bake / export)
  run_nav.py                LIMO waypoint navigation + 세 시점 기록 (Isaac Gym)
  eval_vo.py                사진만으로 SfM → INS 비교
  prepare_robots.py         robot URDF 준비
  select_region.py          overview 지도에서 영역을 골라 LAS로 잘라내기
  build_twin_site.py        잘라낸 색 점군 LAS → digital twin (사진·궤적 없이)
  build_twin_aerial.py      Metashape 항공 사진 project → digital twin (EE2023, configs/blue_wall_aerial.json)
  blender_twin.py           twin을 Blender에서 열어 texture / D455 조정 (docs/blender_tuning.md)
  las_overview.py, crop_las.py   보조 도구 (항공 LiDAR)
umass_lidar/
  resepi.py                 세션 로더, UTM 변환, LAS shift 추정, tile/voxel cache
  camera.py                 카메라 모델, 사진↔LiDAR 정합, calibration
  las_terrain.py            LAS → 2.5D ground / obstacle / roof
  structure.py              3D 구조물 mesh (Poisson)
  texture.py                photo 투영, intensity model, 재질표, GLB export
  metashape.py              Metashape chunk 로더 (pose, depth map → cloud, 항공 사진 bank)
  nav.py                    traversability, A*, pure pursuit
  sim.py, sim_assets.py     Isaac Gym helper (twin 로딩, 조명, robot, 카메라)
  viz.py, mesh.py, io.py    지도 그리기, 2.5D mesh, 보조 도구용 I/O
assets/twin/<name>/         ground.glb(충돌) + <variant>/ tile GLB·PNG·twin_visual.urdf (생성물, git 제외)
data/<name>/                cache: tiles, voxel, depth map, camera.json (git 제외)
outputs/                    calib / vo / nav 결과 (git 제외)
```
