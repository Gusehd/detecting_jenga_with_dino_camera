# jenga-vision: Dino-Lite 젠가 스캔

로봇(오른팔 xArm7 + Allegro 손)이 Dino-Lite 현미경 카메라를 지그에서 집어, 게임 시작 전에 젠가 탑을 한 번 스캔합니다. 블록 끝면 위의 틈(px)으로 "쉽게 빠지는 블록"을 고르고, 뽑을 순서를 JSON으로 동료의 뽑기 코드(`jenga_game`)에 넘깁니다.

## 현재 상태 (2026-11-16)

| 항목 | 값 |
|---|---|
| 기본 실행 | 스캔 v2 (`--preset=v2`, git 태그 `scan-v2`) |
| 전체 시간 | **약 3분 15~20초** (집기 26 s + −y 면 약 76 s + +x 면 약 67 s + 꽂기 26 s). 최선 3:05, 새 탑이면 3:40~4:00 |
| 정확도 | 손으로 밀어 본 정답 기준 **15 / 15 / 20개 연속** (2026-11-10, 탑 2개 3회). −y 면만 쓰면 8~10 |
| 보는 층 | 3~15층 (1·2층과 16·17층은 보지 않음) |
| 아직 안 한 것 | **로봇이 순서대로 실제로 연속 뽑기** (지금 숫자는 살살 밀고 되돌린 결과라 더 낮게 나올 수 있음) |
| 알려진 한계 | 블록의 한쪽 끝(−y 또는 +x)만 봅니다. "한쪽 옆만 열리고 가운데·반대쪽 0"인 층은 구분이 안 될 때가 있습니다 |
| 다음 작업 | 2분대 진입 (사진 회전으로 수평 보정 생략, 두 장 초점 계산 등). 검토: `docs/share_20261116/floor.html` |

자세한 날짜별 기록과 실측값은 `docs/progress_2026-11-02_robot.md` 맨 위 "지금 기준 실행 가이드"에 있습니다. 공유용 보고 페이지는 `docs/share_20261110/index.html`(인수인계 가이드 포함)입니다.

## 설치

```bash
# uv 프로젝트 환경만 씁니다 (시스템 pip 금지). 의존성: numpy, opencv-python, pyusb
~/.local/bin/uv sync
# Dino LED를 끄려면 USB 권한 필요 (한 번만)
sudo cp udev/99-dinolite.rules /etc/udev/rules.d/ && sudo udevadm control --reload
```
ROS 2(Humble) 기능은 `source /opt/ros/humble/setup.bash`를 한 터미널에서 `uv run`으로 씁니다(rclpy는 시스템 ROS에서 옵니다).

## 실행

```bash
cd ~/CES2027/JENGA_VISION_DINO
source tools/robot_env.sh                 # 실기: ROS_DOMAIN_ID=42, USB LAN DDS 설정
ros2 topic hz /joint_states               # 약 30 Hz 나오면 Ctrl+C
uv run jenga-autoscan run --send --preset=v2 --dashboard
# Enter → 집기 → −y 스캔 → +x 스캔 → 순서 저장 → 꽂기. 대시보드 http://127.0.0.1:8765/ (F 전체 화면, L 언어)
# 멈추기: 터미널 Ctrl+C (팔은 그 자리에 섬. 탑 옆에서 관절 이동 금지, 인수인계 가이드의 복구 절차 참고)
```

| 상황 | 명령 |
|---|---|
| Dino를 이미 쥐고 있음 | `run --send --preset=v2 --no-pick --dashboard` |
| 스캔만 (쥔 채로 시작·끝) | `run --send --preset=v2 --no-pick --no-place` |
| 집기 / 꽂기만 | `uv run jenga-task pick_dino_quick --send` / `place_dino_quick --send` |
| 시뮬레이션 (도메인 77) | 터미널 1 `ROS_DOMAIN_ID=77 uv run jenga-sim`, 터미널 2 `ROS_DOMAIN_ID=77 uv run jenga-autoscan run --send --fake --preset=v2` |
| 이전 설정 | `--preset=v1` 또는 `git checkout scan-v1` |
| 채점 | `uv run jenga-autoscan score captures/autoscan_<−y> captures/autoscan_<+x> --truth 3=안빼안 4=빼안빼 ...` |

**현장 준비:** Dino 다이얼 40(스캔 중 돌리지 않기), 팔은 `home`, Dino는 지그에 똑바로 꽂기, `jenga-view`·`jenga-arm`·`jenga-hand` 창 닫기.
**탑 방향(중요):** −y 면에서 볼 때 1층에 블록 끝면(작은 면) 3개가 보이게 쌓습니다. 반대로 쌓으면 −y는 한 층 위를 찍고 +x는 라인을 못 찾아 멈춥니다.

## 통합할 때 알아야 할 점

### 1. 출력 계약 (뽑기 코드가 읽는 파일)
- 스캔이 끝나면 `captures/order_<날짜_시각>.json`을 씁니다. 환경 변수 `JENGA_STATE_DIR`가 있으면 `$JENGA_STATE_DIR/jenga_pick_candidates.json`으로도 복사합니다. 이 경로가 `jenga_game`의 기본 `--candidates`입니다.
- 형식:
  ```json
  {"version": 1, "source": "camera", "stamp": 1794287917.47,
   "candidates": [{"block": "jenga_L07_S2", "score": 26.0, "reason": "빼안빼 rule 1 (gap)"},
                  {"block": "jenga_L07_S0", "score": 26.0, "reason": "빼안빼 rule 2 (opposite)"}, ...]}
  ```
- **블록 이름**은 동료 `jenga_tf`와 같은 `jenga_LXX_SY`입니다. `XX`는 0부터 센 층(L00 = 맨 아래, 즉 사람이 말하는 1층 = L00), `Y`는 슬롯 0/1/2(1 = 가운데)입니다. 슬롯은 탑 자세(`config/scan_setup.json`)로 동료 좌표에 맞춰 바꿔 씁니다. 실물로 한 번 대조해 보는 걸 권합니다.
- **순서가 곧 의미입니다.** 리스트는 위에서부터 뽑을 순서입니다. 층마다 "빼안빼"(양옆이 빠짐) 또는 "안빼안"(가운데가 빠짐)으로 판정합니다. 빼안빼 층은 옆 블록 하나(`gap`) 바로 다음에 반대쪽 옆(`opposite`)이 이어집니다. 가운데 → 옆 순서는 없습니다. `score`는 그 층의 판정 확신도입니다. 가운데 틈과 더 넓은 옆 틈의 차이(px, 절댓값)라서 같은 층 둘은 같은 값입니다. 1 px 이하는 믿기 어려운 판정이고, 리스트 뒤쪽에 옵니다.
- **스캔은 게임 전에 한 번입니다.** 순서는 처음 탑 기준이고, 게임 중에는 다시 재지 않습니다. 사람이 뽑은 블록은 뽑기 코드 쪽에서 후보에서 빼야 합니다. 쉬운 블록만 뺀다면 처음 순위가 게임 내내 유지된다고 보고 이렇게 설계했습니다.
- **파일 전달:** 파일은 이 PC(비전 PC, 192.168.2.33)에 써집니다. `jenga_game`이 로봇 PC(192.168.2.32, 컨테이너 `jenga-ros2`)에서 돈다면 복사 방법이 따로 필요합니다(공유 폴더, scp, 또는 토픽으로 발행). **아직 정하지 않았습니다.**

### 2. 로봇 쪽 사용 자원
- 오른팔과 오른손만 씁니다. MoveIt 없이 `/xarm7_right_controller/follow_joint_trajectory`(관절 궤적, **충돌 검사 없음**)와 `/allegro_hand_position_effort_controller_r/commands`로 직접 보냅니다. 스캔 중에는 다른 노드나 GUI가 오른팔·오른손에 명령하면 안 됩니다.
- 시작과 끝은 팔 `home`, Dino는 지그에 꽂힌 상태입니다(`--no-place`면 쥔 채로 끝남). 뽑기 코드는 이 상태에서 이어받으면 됩니다.
- 자세와 작업은 `config/arm_poses.json`(티칭한 팔 자세), `config/tasks.json`(집기·꽂기 순서와 속도), `config/hand_poses.json`에 있습니다. **덮어쓰기 전에 백업하세요.** 집는 자세 `final_grasp_1_1109_xp5`는 지그 위치에 맞춘 값입니다.
- 탑 위치는 `config/scan_setup.json`에 있습니다(동료의 `tower_pose.yaml`에서 가져온 값과 Dino 보정값). 탑이나 지그를 옮기면 이 값을 다시 맞춰야 합니다.
- ROS 도메인은 **42 = 실기**(로봇 PC는 Jazzy 컨테이너, 이 PC는 Humble), **77 = 시뮬레이션**입니다. `jenga-sim`은 0과 42에서 실행을 거부합니다. 시작할 때 `sequence size exceeds remaining buffer` 경고가 2~4번 뜨는 건 Humble/Jazzy 차이라 무해합니다.

### 3. 시간 예산
- 스캔 한 번이 약 3분 15초입니다. 집기·꽂기 각 26초는 더 빠르면 위험하다고 판단해 고정했습니다.
- 데모 흐름에서 이 시간을 어디에 둘지(관객 앞에서 할지, 미리 할지)는 함께 정해야 합니다.

### 4. AI 도구로 작업할 때
`AGENTS.md`에 규칙을 따로 두었습니다. 핵심은 이렇습니다. 실기 도메인(0, 42)에 움직이는 명령은 사람이 직접 실행하고, 파이썬은 uv 환경만 씁니다.

## 코드 지도

| 파일 | 내용 |
|---|---|
| `src/jenga_vision/autoscan.py` | 로봇 스캔 전체. 설정 묶음 `PRESETS`(v1, v2), 속도 `QUICK`·`FAST`, 순서 규칙 `pick_order`, 출력 `order_main` |
| `src/jenga_vision/ends.py` | 틈 측정 `measure_end` (끝면 위 어두운 띠 폭, 16줄 25 퍼센타일) |
| `src/jenga_vision/focus_search.py` | 초점 탐색 (팔을 광축 방향으로 움직여 가장 선명한 깊이) |
| `src/jenga_vision/task.py` | `jenga-task`: `config/tasks.json`의 팔·손 단계 실행 (집기·꽂기) |
| `src/jenga_vision/sim.py` | `jenga-sim`: 가짜 컨트롤러 + RViz (도메인 77) |
| `src/jenga_vision/dashboard.py`, `dashboard.html` | 데모 대시보드 (Dino 영상 + 픽셀 젠가탑) |
| `src/jenga_vision/camera.py`, `led.py` | Dino 카메라 열기, LED 끄기 (raw USB) |
| `tools/` | 오프라인 분석 스크립트, `robot_env.sh` |
| `docs/` | 날짜별 진행 기록(`progress_*.md`)과 공유 페이지(`share_*/`) |

`captures/`(사진, 세션 결과)는 git에 올리지 않습니다.
