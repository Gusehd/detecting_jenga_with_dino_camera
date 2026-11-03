# 로봇 이식 진행 상황 — 2026-10-29

> **최신은 `progress_2026-10-30_robot.md`.** 결정 배경과 로봇 환경 상세는 `progress_2026-10-28_robot.md`, Dino 순위 알고리즘(`jenga-ends`)은 `progress_2026-10-21.md`, 로봇 토픽/액션은 `../ROBOT_ARM_INTERFACES.md`.

## 한 줄 요약
오른손이 Dino를 거치대에서 **꺼내고(`pick_dino`) 다시 꽂는(`place_dino`) 동작이 실제 하드웨어에서 성공했다**(2026-10-29, 사용자 확인). 다음은 그 사이에 넣을 **탑 1번 면 스캔 모션 티칭**이다.

## 안전 규칙 (꼭 지킬 것)
- **ROS_DOMAIN_ID=0 = 실제 xArm7 + Allegro**(컨트롤러는 다른 PC). Claude는 domain 0에 움직이는 명령을 보내지 않는다. 움직이는 명령은 사용자가 직접 실행한다.
- 모든 도구는 기본이 dry-run이다. 실제로 움직이려면 `--send`를 붙인다.
- Claude가 테스트할 때는 `ROS_DOMAIN_ID=77` + `tools/fake_robot.py`(가짜 팔·손·FK/IK)를 쓴다. 이 스크립트는 domain 0에서는 실행을 거부한다.
- 종료는 Ctrl+C로 한다(Ctrl+Z는 일시정지). 팔이 움직이는 중에 Ctrl+C를 누르면 이동이 취소되고 팔은 그 자리에 선다. 가짜 컨트롤러로 취소가 도착하는 것을 확인했다.

## 도구 (모두 ROS를 source한 터미널에서 `uv run ...`)
| 명령 | 내용 |
|---|---|
| `jenga-task [이름] [--send] [--step] [--from N] [--arm-speed 10] [--hand-speed 20]` | `config/tasks.json`의 단계(팔 자세 / 손 자세 / 손 시퀀스)를 차례로 실행한다. 이름 없이 실행하면 목록을 보여 준다. 단계별 `"speed"`로 속도를 따로 줄 수 있다 |
| `jenga-arm --arm right [--send]` | xArm7 7관절 슬라이더 GUI. **슬라이더는 실제 관절값을 따라가다가, 만지면 멈춘다**(GO 또는 Robot→sliders로 다시 따라감). **Save robot pose** = 실제 관절값 저장(`config/arm_poses.json`). **Cartesian jog** 패널: hand/base 축 기준 X/Y/Z mm, Rx/Ry/Rz deg 이동(MoveIt `/compute_ik`, 1 mm/1° 단위, 관절이 5° 넘게 튀면 거부) |
| `jenga-route teach <경로> --arm right` | 매뉴얼 모드/조그로 옮긴 실제 자세를 Enter(촬영 자세) / `v`(경유 자세)로 순서대로 저장한다. `i N` 끼워 넣기, `r N` 덮어쓰기, `d N` 삭제, `l` 목록. 로봇에 아무것도 보내지 않는다. `config/arm_routes.json` |
| `jenga-route play <경로> --arm right [--send] [--speed 5] [--from N --to M] [--auto]` | 경로를 순서대로 재생한다(관절 공간). 기본은 이동마다 Enter를 기다리고, 도착하면 관절 오차를 출력한다 |
| `jenga-hand --hand r [--send]` | Allegro 20관절 슬라이더, 자세 저장, 시퀀스 Run. ⚠️ `jenga-task`로 손을 움직인 뒤에는 이 GUI의 명령값이 실제와 다르다. 끄거나 Sync 후 사용 |
| `jenga-view --ends` | Dino 화면 + 하늘색 가운데 선 + 실시간 틈(px) + focus. `o` LED, `s` 저장 |
| `jenga-led on/off` | Dino LED. **이제 모든 Dino 도구가 카메라를 열 때 LED를 끈다**(기본 OFF) |

## 저장된 것 (config/)
- 팔(오른쪽) `arm_poses.json`: `home`, `before_grasp`, `final_grasp_1`, `final_grasp_1_1`, `final_grasp_1_2` (작업에서 사용) + `pose_1`, `pose_2`, `grasp_ready`, `grasp_ready_1`, `final_grasp_2~4` (안 씀)
- 손(r) `hand_poses.json`: `initial`, `initial_1`, `grasp_dino`, `grasp_middle_1`
- 손 시퀀스 `hand_sequences.json`: `grasp_middle_1` = 새끼 → 약지 → 중지 → 검지 → 엄지, `release_middle_1` = 역순으로 `initial_1` (`grasp_dino` / `release_dino`는 예전 것)
- 작업 `tasks.json`
  - `pick_dino`: 팔 home → 손 initial_1 → 팔 before_grasp → final_grasp_1 → 손 grasp_middle_1 → 팔 final_grasp_1_1 → final_grasp_1_2 → before_grasp → home
  - `place_dino`: 위의 역순(손은 release_middle_1)
- 경로 `arm_routes.json`: `f1`(2점), `f2`(3점) = 도구 시험용. 지우거나 덮어써도 된다

참고: before_grasp → final_grasp_1에서 j7이 139°, j3이 75° 돈다(팔꿈치 구성이 다름). 실제로는 문제없이 동작했지만 거치대 위치가 바뀌면 다시 확인한다.

## 진행 단계
| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 심에서 손·팔 GUI 확인 | ✅ |
| 1 | 하드웨어에서 Dino 쥐기 | ✅ |
| 1.5 | 거치대에서 꺼내기/꽂기 작업 (`pick_dino` / `place_dino`) | ✅ 2026-10-29 |
| **2** | **오른팔 1번 면 스캔 모션 티칭** (아래 가이드) | **← 다음** |
| 3 | 재현성·흔들림·미끄러짐 확인 (같은 경로를 여러 번 재생) | |
| 4 | 스캔 스크립트: 경로 재생 + 촬영 자세마다 정지 대기 → 촬영 → `ends.measure_end` → 저장 | |
| 5 | `jenga-task`에 스캔 단계 추가: pick → 1번 면 스캔 → place 를 한 번에 | |
| 6 | 2번 면(오른팔), 3·4번 면(왼팔 + 왼손 pick/place 티칭), 가운데 거치대로 Dino 넘기기 | |
| 7 | 4면 결과 → `jenga-ends` 점수 → 순위 JSON 저장 | |
| 8 | 부드럽게: 연속된 팔 이동을 궤적 하나로 합치기(경유점에서 멈추지 않음), 팔·손 동시 동작 | |

## 2단계 가이드: 스캔 모션 짚어내기

### 무엇을 찍나
- 한 면에서는 **한 층 건너** 끝면이 보인다. 1번 면 = `--first` 층(L02)이 끝면 3개를 보이는 쪽.
  - 1번 면: L02, L04, …, L16 = 8층 × 3블록(left/middle/right) = **24장**
  - 2번 면(90° 돌린 쪽): L03, L05, …, L15 = 7층 × 3 = 21장. 3·4번 면은 반대편이다. 4면 합계 90장(`jenga-ends`와 같은 구성)
- 블록 이름은 그 층의 첫 면에서 본 left/middle/right다(먼 면에서는 화면 왼쪽이 그 층의 right). 자세 이름에 `L02 left`처럼 층과 블록을 적어 둔다.

### 준비 (터미널 3개)
1. `uv run jenga-task pick_dino --send` → Dino를 쥔 채 home에 온다.
2. `uv run jenga-view --ends`: LED는 자동으로 꺼진다. 하늘색 선, 틈 px, focus 확인용.
3. `uv run jenga-arm --arm right --send`: Cartesian jog로 미세 조정. **frame = hand**, step 1~2 mm로 시작해서 마지막엔 0.5 mm.
4. `uv run jenga-route teach scan_f1 --arm right`: 자세를 찍는다.
- 큰 이동은 매뉴얼 모드(손으로)로 하고, 미세 조정은 조그로 한다. 둘 다 실제 관절값이 저장된다.

### 촬영 자세 하나를 맞추는 기준
1. 끝면 가운데가 **하늘색 세로선**에 오게 한다.
2. 틈(끝면 윗변과 위 블록 아랫변)이 **수평**이 되게 한다(필요하면 Rz 조그).
3. **focus가 최대**인 거리로 맞춘다. 광축 방향으로 앞뒤 0.5 mm씩 움직이며 값이 가장 높은 곳을 찾는다.
   - focus 절대값에는 최소 기준이 없다. 예전 성공 사진은 320~840(중간값 약 550)이었고, 지금 약 1000이 나와서 충분하다. 같은 자리에서 비교하는 용도로만 쓴다.
   - hand 축 중 어느 것이 Dino 광축인지 먼저 확인한다(X/Y/Z 조그를 1 mm씩 눌러 보고 focus가 변하는 축).
4. **배율 확인**: 뜬 블록의 틈이 예전처럼 10~15 px 이상 보여야 한다(배율 50에서 틈 중간값 6 px, 큰 틈 15 px 정도). 몇 px밖에 안 나오면 배율이 낮은 것이니 더 가까이 간다.
5. **탑에 닿지 않게** 한다. 모든 자세에서 거리를 같게 유지한다(px 크기 비교가 공정해야 함).

### 찍는 순서 (추천)
1. `v scan_f1_approach`: 탑에서 5 cm 이상 떨어진 안전 자세(경유)
2. **우선 한 층만 찍는다**: `L02 left`, `L02 middle`, `L02 right`(촬영 자세)
3. 층을 옮기기 전에 탑에서 살짝 뒤로 뺀 경유 자세(`v`)를 넣는다. 같은 층 블록 사이는 경유 자세가 없어도 된다.
4. `v scan_f1_retreat`: 다 찍고 빠지는 자세
5. `uv run jenga-route play scan_f1 --arm right --send --speed 3` 로 천천히 재생한다. 경로가 탑에 가까우면 Ctrl+C로 멈추고 `teach`에서 `i N` → `v`로 경유 자세를 끼워 넣는다.

### 한 층이 되면: 나머지를 어떻게 채울지 정한다
- **A. 24장을 전부 손으로 찍기**: 계산이 필요 없고 가장 확실하다. 시간은 걸린다.
- **B. 몇 개만 찍고 계산하기**: `L02 left`, `L02 middle`, `L04 left`, `L16 left`를 찍는다. Claude가 `/compute_fk`로 `dx`(옆 블록), `dz`(2층 위), 탑 기울기를 구하고, 나머지는 오프셋 → `/compute_ik`(찍은 관절값을 시작값으로)로 만든다. `cartesian.py`의 `Cartesian.fk/ik`를 그대로 쓰면 된다.
- 판단 기준은 3단계 결과다. 같은 경로를 3번 재생했을 때 틈 위치가 몇 px 흔들리는지 본다. 작으면 B로 충분하고, 크면 A로 찍거나 영상으로 위치를 보정해야 한다.

## 다음 세션에서 Claude가 할 일
- 사용자가 `scan_f1`(최소 한 층)을 찍어 오면 다음을 한다.
  1. `jenga-route list --arm right scan_f1`로 확인한다. 촬영 자세 사이의 관절 변화량을 보고 이상한 큰 회전이 있는지 본다.
  2. **스캔 스크립트(4단계)**: 경로 재생 + 촬영 자세에서 정지 대기(프레임 간 이동량이 작아질 때까지) → `ends.measure_end` → 세션 폴더에 사진 + `ends.csv` 저장. 기존 `jenga-ends`의 파일 형식(`L02:f1:left` 키)을 맞춰서 `ends.py`의 점수/순위 함수를 그대로 쓴다. 기본은 dry-run.
  3. B 방식이면 `dx`/`dz` 계산 코드를 만든다.
  4. `jenga-task`에 `{"scan": "scan_f1"}` 단계 종류를 추가해서 pick → scan → place를 한 번에 돌린다.
- 카메라(`/dev/video0`)는 프로세스 하나만 열 수 있다. 스캔 스크립트를 돌릴 때는 `jenga-view`를 끈다.

## 알아 둘 것
- **ROS 연결이 끊기는 문제(2026-10-28)**: 이 PC는 Wi-Fi(`wlo1`)로 로봇 PC에 붙는다. 한 번 `/joint_states`가 안 들어온 적이 있다(`ros2 node list --no-daemon`이 비어 있었음). `ros2 topic list`는 데몬에 남은 예전 목록을 보여 주니 믿지 말고, `ros2 daemon stop` 후 `ros2 topic hz /joint_states`로 확인한다. 가능하면 유선으로 연결한다.
- `jenga-arm`의 회전 조그는 손 기준점(`ahr_base_link` 원점)을 중심으로 돈다. Dino 끝을 중심으로 돌리려면 손 기준으로 Dino 끝까지의 거리를 재서 TCP 오프셋으로 넣는다(미구현).
- Cartesian jog와 FK/IK는 **가짜 MoveIt으로만 테스트했다.** 실제 로봇 PC의 move_group으로는 아직 확인하지 않았다. 처음 쓸 때 dry-run으로 패널의 손 위치가 나오는지, "IK ok"가 뜨는지 먼저 본다.
- 오늘 바뀐 코드: LED 기본 OFF(`camera.py`, 각 도구), `arm_route.py`, `cartesian.py`, `task.py`, `arm_slider.py`(슬라이더 따라가기, 조그, Ctrl+C 취소 수정, 공유 노드), `hand_slider.py`(공유 노드), `tools/fake_robot.py`.

## 추가 (2026-10-29 저녁): 팀원 URDF로 자동 스캔 계획 초안
팀원에게 받은 `jenga_world/`(jenga_bringup + jenga_sw, 각자 .git이 있어 이 저장소에는 커밋하지 않는다)를 참고했다.

### 알아낸 것
- **탑**(팀원 `tower_pose.yaml`, xacro의 stand): 월드 (0.40, 0.0), yaw −45°, 바닥 z=0.101. 블록 115 × 38 mm, 층 높이 22.36 mm, 층 L00(맨 아래)~L16, 슬롯 S0~S2(S1 = 가운데). 슬롯 간격 38.5 mm.
- 짝수 층은 끝면이 탑 ±y 면, 홀수 층은 ±x 면에 있다. **오른팔 = −y, +x / 왼팔 = −x, +y**로 나누면 각 팔이 모든 블록의 한쪽 끝을 하나씩 본다(45개씩). 두 팔 결과를 합치면 예전에 가장 좋았던 `mean`(양 끝 평균)을 그대로 쓸 수 있다.
- 층 번호: 팀원 L00 = 맨 아래 = 예전 `jenga-ends`의 L01. 스캔 범위 L01~L15(팀원) = 예전 L02~L16.
- **FK 검증**: 직접 만든 URDF FK(`kinematics.py`)로 계산하면 `final_grasp_1` → `_1_1` → `_1_2`가 월드 z로 정확히 +40.0, +50.0 mm이고 손 방향은 그대로다. 실제 로봇과 맞는다는 근거다. Dino는 지그에 거꾸로(끝이 아래로) 꽂혀 있고 손 좌표로 Dino 끝 방향 ≈ (−0.017, −0.961, −0.275)다.
- **거울 대칭**: 오른팔 관절값 × [−1, 1, −1, 1, −1, 1, −1] = 왼팔의 정확한 거울 자세(월드 y=0 기준, 오차 0)다.

### 만든 것
| 명령/파일 | 내용 |
|---|---|
| `src/jenga_vision/kinematics.py` | URDF에서 바로 FK/IK(ROS 없이). 7관절 DLS IK, 시작값 근처의 해 |
| `uv run python -m jenga_vision.scan_plan [--standoff 0.04] [--write]` | `config/scan_setup.json`(탑, Dino 추정치, 면 배정)으로 면마다 촬영 자세를 IK로 계획한다. 점 사이는 5 mm 단위로 따라가서 팔 자세가 이어지게 했고, 관절 제약(팀원 기준 j4 ≥ 10°, \|j1\| ≤ 120°)과 home ↔ 면 이동 중 탑과의 거리를 검사한다. `--write`면 경로와 왼팔 자세를 저장한다. 그림: `captures/scan_plan.png` |
| 경로 `auto_scan_right_my`(−y, 21장), `auto_scan_right_px`(+x, 24장), `auto_scan_left_mx`(−x, 24장), `auto_scan_left_py`(+y, 21장) | **탑에서 40 mm 떨어진 초안**. 블록마다 촬영 자세가 있고, 층을 바꿀 때 30 mm 빠지고, 처음과 끝에 200 mm 먼 경유점이 있다 |
| 왼팔 `auto_home`, `auto_before_grasp`, `auto_final_grasp_1(_1, _2)` | 오른팔 자세의 거울 대칭(**지그가 y=0이라고 가정**) |
| 왼손 `initial_1`, `grasp_middle_1` | 오른손 값 복사(왼손에서 같은 값이 거울 자세가 되는지 미확인). 원래 파일은 `config/hand_poses.json.bak2` |
| 작업 `scan_right_draft`, `scan_left_draft` | 꺼내기 → 두 면 이동 → home → 꽂기(촬영 없이 동작만). 팔 이동만 각 약 6분. `pick_dino_left`, `place_dino_left`도 있다 |
| `jenga-task` | 새 단계 종류 `{"route": 이름}`(경로 재생), `{"task": 이름}`(다른 작업 끼워 넣기) |

### 계산으로 정한 것
- **Dino 회전(손이 Dino 축 둘레 어디에 있나)**: 오른팔 270°, 왼팔 90°일 때 네 면 모두 도달 가능하다(0/90/180/270°를 모두 계산했다). 화면은 그만큼 돌아가므로 촬영 사진을 소프트웨어에서 90° 단위로 돌린다(`image_rotate_deg`, 첫 실제 촬영에서 정한다).
- 3 mm 거리에서는 같은 층 옆 블록으로 옮길 때 관절 보간 경로가 탑 쪽으로 최대 1.4 mm 처진다. 탑 위치 오차(팀원 문서: x/y 6 mm)까지 생각하면 3 mm는 위험하다. 그래서 초안은 40 mm다.

### 실제로 돌려 볼 순서 (사용자)
1. `uv run jenga-task scan_right_draft` (dry-run)로 단계를 확인한다.
2. `uv run jenga-task scan_right_draft --send --step --arm-speed 5`로 실행한다. 특히 home → 면의 큰 이동(관절 최대 150°)을 지켜본다.
3. 촬영 자세에서 Dino가 블록 끝면 쪽을 대략 향하고 있는지 본다(40 mm 거리라 초점은 안 맞는다). 이 결과로 Dino 끝 위치 추정치를 보정한다.
4. 왼팔은 지그 위치를 확인한 뒤에 돌린다(아래 요청 목록 1번).

### 정확하게 만들려면 필요한 것
**사용자(측정/티칭)**
1. **지그 중심의 월드 y**(로봇 정중앙에서 몇 mm). 0이 아니면 `scan_setup.json`의 `jig_y`에 넣고 `--write`를 다시 실행한다. 또는 왼팔 쥐기 자세를 오른팔처럼 직접 티칭한다.
2. **보정용 촬영 자세 1개**(오른팔 −y 면 아무 블록, 초점·가운데 맞춤). 이것으로 Dino 끝 위치(손 기준)를 계산하면 나머지 89장은 계산으로 맞출 수 있다. 가능하면 면마다 1개(4개)가 좋다(탑 위치 오차도 같이 보정된다).
3. 왼손으로 같은 관절값의 `grasp_middle_1`이 거울 자세가 되는지 확인한다.

**팀원에게**
1. **최신 URDF**: 받은 `dual_robot.urdf`는 10/15 것이라 탑 받침대, TCP 링크가 없다. 그리고 손 장착 각도가 xacro와 다르다(URDF rpy 0, xacro rpy 0 0 ±90°). 지금 로봇에 올라가 있는 것(`/robot_description`)으로 받기. 왼팔의 D405 판(5 mm)이 실제로 붙어 있는지도.
2. **탑 위치가 지금 실제와 맞는지**: `tower_pose.yaml`(0.40, 0, −45°, z 0.101)이 현재 지그 위치인지, 오차가 얼마인지(문서상 x/y 6 mm).
3. **지그(Dino 거치대) 위치**를 팀원 씬/충돌 모델에 넣을지, 이미 위치를 아는지.
4. 게임 중 양팔의 **대기 자세(home)**: 스캔 동작과 겹치지 않게.
5. Allegro **왼손과 오른손 관절 부호**가 같은 값이면 거울 자세인지.
