# 로봇 이식 진행 상황 — 2026-10-28

> ⚠️ 최신 상태는 `progress_2026-10-29_robot.md`(새 세션은 그 문서부터). 이 문서는 결정 배경과 로봇 환경 기록이다.
> 원래 안내: 새 세션은 이 문서부터 읽는다. Dino 순위 알고리즘(`jenga-ends`)은 `progress_2026-10-21.md`, 로봇 토픽/액션은 `../ROBOT_ARM_INTERFACES.md`.

## 결정된 것
- **손목 고정 마운트는 버림.** 손목에 붙이면 초점거리 약 20 cm = Dino 최저 배율이다. 예전 촬영(배율 50, 캡 붙임, 약 0.012 mm/px)보다 10~20배 거칠어서 틈 차이(0.05~0.25 mm)가 1 px 아래로 묻힌다. (자로 mm/px를 실측하지는 않았다.)
- **오른손 Allegro가 Dino를 쥐고 가까이 가져간다.** Dino 윗면이 손바닥 쪽으로 오게 쥔다. Dino를 꺼내는 거치대/3D 프린팅은 일단 해결된 것으로 가정한다.
- 왼팔이 2면, 오른팔이 2면을 맡는다. 가운데 거치대에 Dino를 꽂아서 넘긴다. 탑은 지그로 고정한다(위치를 매번 찾지 않음).
- **모션은 전부 룰베이스.** 면마다 시작 자세를 티칭하고, 나머지는 `dx`(옆 블록) / `dz`(2층 위; 한 면에서는 한 층 건너 끝면이 보임) 오프셋을 누적해서 만든다.
- **최종 목표:** Python 스크립트 하나로 Dino 쥐기 → 4면 스캔 → 기존 `jenga-ends` 점수 → 순위를 JSON으로 저장.

## 로봇 환경
- ROS 2 Humble, **ROS_DOMAIN_ID=0 = 실제 하드웨어**에 연결된다(컨트롤러는 다른 PC). **움직이는 명령은 사용자가 직접 실행한다.** 도구는 기본이 dry-run이다.
- 팔: `/xarm7_{left,right}_controller/follow_joint_trajectory`(액션, 완료 결과가 옴). MoveIt은 경로 계획 없이 `/compute_fk`, `/compute_ik`(계산만)만 쓸 예정이다.
- 손: `/allegro_hand_position_effort_controller_{l,r}/commands`, Float64MultiArray, 20개 값(엄지 0~3, 검지, 중지, 약지, 새끼 순). 심과 하드웨어 모두 20개로 동작했다.
- `/joint_states`: `ahl_/ahr_joint00..43`, `xarm_{left,right}_joint1..7`.

## 만든 도구 (모두 `uv run`, ROS를 source한 상태에서)
| 명령 | 내용 |
|---|---|
| `jenga-hand --hand r [--send] [--speed 15]` | Allegro 20관절 슬라이더. 자세 저장/Go(부드럽게), **시퀀스 Run**, STOP. 자세는 `config/hand_poses.json`, 시퀀스는 `config/hand_sequences.json` |
| `jenga-arm --arm right [--send] [--speed 10]` | xArm7 7관절 슬라이더. GO를 눌러야 움직임. 이동 시간은 속도 제한으로 계산(최소 2 s). **Save robot pose** = 티칭. `config/arm_poses.json` |
| `jenga-arm` Cartesian jog 패널 | 손(`ahr_base_link`)을 hand/base 축 기준으로 X/Y/Z mm, Rx/Ry/Rz deg씩 이동. MoveIt `/compute_fk`·`/compute_ik`로 1 mm / 1° 단위 IK(앞 해를 시작값으로), 관절이 한 단계에 5° 넘게 튀면 거부. `--send`면 실제 자세에서 출발해 바로 이동, dry-run이면 슬라이더만 바뀜 |
| `jenga-task [pick_dino\|place_dino] [--send] [--step] [--from N] [--arm-speed 10] [--hand-speed 20]` | `config/tasks.json`의 단계(팔 자세 / 손 자세 / 손 시퀀스)를 차례로 실행. 이름 없이 실행하면 목록. 기본은 dry-run(이동 시간 출력). Ctrl+C = 팔 이동 취소, 손 명령 정지. 실행 중에는 `jenga-hand --send`를 끄거나 나중에 Sync |
| `jenga-route teach f1 --arm right` | 매뉴얼 모드로 팔을 옮기고 Enter(검사 자세) / `v`(경유 자세)로 관절값 저장. `i N` = N번 뒤에 끼워 넣기, `r N` 덮어쓰기, `d N` 삭제. 아무것도 보내지 않음. `config/arm_routes.json` |
| `jenga-route play f1 --arm right [--send] [--speed 5] [--from N --to M] [--auto]` | 저장 순서대로 관절 공간 이동. 기본은 매 이동 전 Enter 대기. Ctrl+C = 이동 취소(팔 정지) |
| `jenga-view --ends` | Dino 화면 + 하늘색 가운데 선 + 실시간 틈(px). `e` 오버레이, `s` 저장, `o` LED |

- Dino LED는 카메라를 열 때 기본으로 꺼진다(`jenga-view`에서 `o`로 켬).
- Ctrl+C로 종료한다. **Ctrl+Z는 일시정지**라 쓰지 않는다.
- 저장된 손 자세(오른손): `initial`(편 손), `grasp_dino`.
- 시퀀스: `grasp_middle_1` = 새끼 → 약지 → 중지 → 검지 → 엄지, `release_middle_1` = 역순으로 `initial_1`.
- 작업(2026-10-29): `pick_dino` = 팔 home → 손 initial_1 → 팔 before_grasp → final_grasp_1 → 손 grasp_middle_1 → 팔 final_grasp_1_1 → final_grasp_1_2 → before_grasp → home. `place_dino` = 그 역순.
- 시퀀스: `grasp_dino` = 새끼 → 약지 → 중지 → 엄지 → 검지, `release_dino` = 그 역순으로 `initial`.
- 실제 하드웨어에서 쥐기가 잘 되는 것을 확인했다(사용자).
- 주의: `grasp_dino`는 엄지 4관절과 중지·약지·새끼 j1~j3이 한계값이다. 오래 쥐면 모터가 열 받을 수 있으니 접촉 지점 근처로 줄이는 것을 고려한다.

## 진행 단계
| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 심에서 손·팔 GUI 확인 | ✅ |
| 1 | 하드웨어에서 Dino 쥐기 자세 + 시퀀스 | ✅ |
| **2** | **팔 티칭 (오른팔, 1번 면)** | **← 다음** |
| 3 | FK로 `dx`, `dz` 계산 → 한 면 자동 모션 (오프셋 → IK, 티칭 관절값을 시작값으로) | |
| 4 | 한 면 모션으로 흔들림(settling) / 미끄러짐(특히 축 방향 회전) 확인 | |
| 5 | 4면 + 양팔 Dino 넘기기 + 순위 JSON | |

### 2단계 방법
(2026-10-28 변경) 슬라이더 대신 `jenga-route teach`로 관절값을 찍고 `jenga-route play`로 느리게 재생한다. 경로가 탑에 가까우면 그 사이에 경유 자세(`i N` 후 `v`)를 넣는다.

터미널 3개: `jenga-hand --hand r --send`(쥔 상태 유지), `jenga-view --ends`, `jenga-arm --arm right`(저장만 하면 되므로 dry-run). 먼저 `jenga-led off`.
매뉴얼 모드로 팔을 옮기면서 맞춘다. 끝면 가운데를 하늘색 선에, 틈을 수평으로, `focus` 값이 최대인 거리에서, 탑에 닿지 않게(배율 50부터).

저장할 자세 (Save robot pose):
| 이름 | 위치 |
|---|---|
| `f1_approach` | 탑에서 떨어진 안전 자세 |
| `f1_L02_left` | 2층 왼쪽 끝면 (시작점) |
| `f1_L02_middle` | 2층 가운데 끝면 (`dx`) |
| `f1_L04_left` | 4층 왼쪽 끝면 (`dz`) |
| `f1_L16_left` | 16층 왼쪽 끝면 (누적 오차 확인) |

### 다음 세션에서 할 일 (Claude)
- 저장된 관절값을 `/compute_fk`에 넣어 `dx`, `dz`, 탑 기울기를 계산한다.
- 한 면 스캔 스크립트: 자세마다 오프셋 → `/compute_ik` → `follow_joint_trajectory` → settling 대기(프레임 간 이동량으로 판단) → 촬영 → 기존 `ends.measure_end`. 기본은 dry-run.
- 미세 조정이 불편하다면 손끝 X/Y/Z mm 조그(IK)를 `jenga-arm`에 추가한다.
