# 로봇 이식 진행 상황 — 2026-10-30

> **새 세션은 이 문서부터 읽는다.** 전날 상황(pick/place 성공, 자동 경로 초안)은 `progress_2026-10-29_robot.md`, 로봇 환경과 결정 배경은 `progress_2026-10-28_robot.md`, 순위 알고리즘(`jenga-ends`)은 `progress_2026-10-21.md`.

## 한 줄 요약
Dino **보정**(손 기준 초점점)을 끝냈고, 실제 로봇에서 **손 장착 각도 오류**(URDF 0° → 실제 +90°)를 찾아 고쳤다. **자동 스캔 `jenga-autoscan`**(초점 탐색 → 틈 찾기 → 수평 → 촬영 → 순위 JSON)을 만들었다. 가상 시험은 통과했지만 **실기에서는 아직 한 장도 제대로 찍지 못했다.** 마지막 문제(배율 30에서 초점 최고점이 출발점보다 바깥)는 코드로 고쳤고 다음에 실기로 확인한다.

## 목표 다시 정리 (사용자 확인)
- **새로 쌓은 탑을 게임 전에 딱 한 번** 스캔한다. 블록마다 끝면 윗 모서리의 틈(px)을 재서 1등~꼴등 순위를 만든다(기존 `jenga-ends` 알고리즘).
- 결과는 **순위 JSON**(팀원 블록 이름 `jenga_LXX_SY`)으로 넘긴다. 팀원 코드가 그 순서대로 뽑는다.
- 탑은 쌓을 때마다 mm 단위로 다르니(깊이 최대 10 mm, 기울기), 고정 경로가 아니라 **대략 위치 + 영상으로 찾기**로 간다.

## 안전 규칙 (그대로 + 추가)
- ROS_DOMAIN_ID=0 = 실제 로봇. Claude는 움직이는 명령을 보내지 않는다. 시험은 domain 77 + `jenga-sim`.
- **배율 50에서는 초점이 Dino 끝이 블록에 거의 닿아야 맞는다.** 자동 접근으로 블록을 밀 뻔했다(L13 왼쪽). 그래서 **다이얼을 30으로 낮췄다**. 30에서는 끝과 블록 사이에 여유가 있다. 스캔하는 동안 다이얼은 고정한다.
- 거치대에서 Dino를 다시 꺼낼 때마다 손 안에서 **축 방향으로 1 cm 넘게, 축 둘레로 약 5°** 달라질 수 있다. 깊이와 회전은 매 실행마다 영상으로 다시 찾는다.
- 탑 근처에 서 있는 팔에서 `pick_dino` / `place_dino` / `--via` 관절 이동을 바로 시작하지 않는다(관절 보간이 탑을 쓸 수 있음). 먼저 광축으로 빼거나 매뉴얼로 10 cm 이상 뺀다.
- Claude 셸에서는 domain 0 토픽이 안 보인다(샌드박스를 꺼도 마찬가지). 실제 로봇 값은 사용자가 출력을 붙여 준다.

## 오늘 알게 된 것
| 항목 | 값 / 결론 | 근거 |
|---|---|---|
| 보정 자세 | `cali_way_3` = **+x 면 L01_S0 윗 모서리**(사용자 표현 "아래에서 두 번째 층, 왼쪽 블록 s1") | 사진 `captures/20261030_115917_128.png` |
| 블록 확인 | 거치대 Dino → 탑 지그 가까운 꼭짓점 **205 mm 실측**, S0이면 202 mm 예측(S1 163, S2 125) | 줄자 |
| 거치대 위치 | 월드 y ≈ 0 (로봇 중앙선) → 왼팔 거울 대칭 가정 OK | 보정값으로 FK |
| **손 장착 각도** | **실제 = xacro의 z ±90°**(오른손 +1.5708, 왼손 −1.5708). 받은 10/15 URDF는 0° | dino 프레임 −Z 조그가 "아래로" 감 → 수정 후 정상 |
| 층 높이 | **21.86 mm**(팀원 값 22.36). L13에서 6 mm 차이 → 고친 뒤 −y 면 L14에서도 1.4 mm 이내 | `s13_*`, `s14_*` |
| 탑 깊이 | 모델보다 위층 끝면이 **7~10 mm 안쪽**(+x L13: +9.6~9.9, −y L14: +7.4~8.7) | 손으로 맞춘 자세와 계획 비교 |
| 탑 높이 | 바닥 z = **0.101**(지그를 하나 더 깔아서 높음, 팀원 문서의 0.051은 옛 값) | 사용자 |
| focus 값 | `jenga-view`의 라플라시안 분산은 노이즈에 묻힘(맞음 741~1619 / 흐림 293~572). **가우시안 차이(σ1−σ3)**로 바꿈: 맞음 44~135 / 흐림 2~4 (배율 50 사진) | 실제 사진 9장 |
| 모서리 구분 | 윗 모서리 = 틈 **위가 가로 줄무늬**(위층 블록 옆면), 아래가 점 같은 끝면. 줄무늬 지표 위 1.10~1.66 / 아래 0.95~1.13 (6장 모두 맞음, 단 차이가 작은 사진도 있음) | 배율 50 사진 |
| 배율 30 초점 곡선 | L13 왼쪽: −4 mm에서 35(최고), 들어갈수록 계속 떨어짐(+13 mm에서 3.8) → **초점점이 보정값보다 바깥으로 이동**(최소 4 mm 이상) | `captures/autoscan_20261030_163402/focus_log.csv` |

## 만든 것 / 바꾼 것
| 명령 / 파일 | 내용 |
|---|---|
| `jenga-sim` (`sim.py`, `config/sim.rviz`) | domain 77 가상 로봇: 양팔·양손 컨트롤러, URDF FK/IK 서비스, RViz 마커(탑, 팔 뼈대, 손가락, Dino, 촬영 지점), 탑과 5 mm 이내면 경고. `--check TASK`: ROS 없이 작업 전체의 탑까지 거리 |
| `jenga-calib` (`calib.py`) | 손으로 맞춘 촬영 자세 1개 + 블록 번호 → `scan_setup.json`의 `dino.tool_hand`(원점 = 초점점, z = 광축, y = 화면 아래) |
| `jenga-autoscan` (`autoscan.py`) | 자동 스캔. 아래 절 참고 |
| `jenga-arm` | Cartesian jog에 **frame `dino`** 추가: Z = 광축, X/Y = 화면 오른쪽/아래, 회전은 초점점 중심 |
| `jenga-task` | `--step`이면 **경로의 점마다** Enter를 기다림 |
| `scan_plan.py` | 보정된 tool 사용(`arm_tool`), 겨누는 곳 = 끝면 **윗 모서리**(`aim: 1.0`), `--only ARM FACE`(px/mx/py/my), `--layers`, `--seed`, `--name`, 관절이 ±360° 한계에 붙으면 360° 옮김 |
| `kinematics.py` | `RPY_OVERRIDE`: 손 장착 각도를 실제 로봇 값으로 |
| `config/scan_setup.json` | `tool_hand.right`(cali_way_3), `aim 1.0`, `layer_h 0.02186`, `standoff_m 0`, `tip_to_focus_m 0.003`(추정, sim 거리 검사용) |
| `config/arm_poses.json` (right) | `cali_way_1`(경유), `cali_way_2/3`(보정), `s13_1~3`(+x L13 손으로 맞춤), `s14_1~3`(−y L14) |
| `config/arm_routes.json` / `tasks.json` | 검증용 `verify_px`(+x L13, 5 mm 바깥), `verify_my`(−y L14, 15 mm 바깥, home→면에서 j5 약 180° 회전) |

백업: Claude 세션 scratchpad에 `scan_setup.before_calib.json`, `scan_setup.before_layerh.json`, `arm_routes.before_mount.json`(세션이 끝나면 사라질 수 있음. 필요한 값은 이 문서에 있음).

## `jenga-autoscan` 동작 (면 하나)
1. `--via` 자세들 → 면 앞 먼 점(`scan_plan`으로 계산, `--seed` 자세의 팔 구성) → 층마다 왼쪽 블록 30 mm 앞.
2. **초점 찾기**: 예상 깊이 − `--start-mm`(8)에서 0.5 mm씩 들어감. 최고점을 지나 2번 연속 90% 아래로 떨어지면 멈추고 0.125 mm로 다듬음. **처음부터 3번 연속 떨어지기만 하면 방향을 바꿔 바깥으로**(최대 15 mm) 찾음. 한계: 직전 블록 깊이 + `--limit-mm`(2), 전체 `--max-in-mm`(25).
3. **틈 찾기**(면에서 처음 한 번): 화면 1/3 구간 3개 중 2개에서 어두운 띠가 보이면 틈. 없으면 위아래 2.5 mm 간격으로 ±`--gap-search-mm`(12.5), 위쪽 먼저.
4. **윗/아랫 모서리 판정**(나뭇결). 아랫 모서리면 층 높이만큼 올라가 다시 찾음.
5. **틈 가운데 맞추기**(첫 장에서 Y 0.5 mm 움직여 µm/px 측정) → **수평 맞추기**(좌우 1/3의 틈 높이 차이로 Dino 축 회전, 부호는 영상으로 확인) → 다시 가운데 → ±0.5 mm 재초점.
6. **촬영**: 3장, `measure_end` 중간값. 옆 블록은 모델 위치 + 찾은 깊이/높이/회전에서 ±3 mm 재초점.
7. 결과 `captures/autoscan_<시간>/`: `ends.csv`(jenga-ends 키 `L<층+1>:f<면>:<화면 위치>`, 면 1~4 = +x, +y, −x, −y), `shots.csv`(깊이, 높이, 회전, 관절값), `focus_log.csv`(모든 초점 측정), 사진, `ranking.json`.
8. 여러 세션 합치기: `uv run jenga-autoscan rank 세션1 세션2 --out captures/ranking.json`.
- `--send` 없이: ROS 없이 가상 팔 + 가짜 카메라(깊이 오차, 기울기, 높이 오차, Dino 회전, 아랫 모서리까지 합성). `--send --fake`: domain 77 `jenga-sim`으로 RViz에서 확인.
- `--send`면 출발 전에 Enter 확인, 카메라 창이 켜짐. 창에서 **스페이스바 = 지금 깊이로 정함, q = 중단**.
- 가상 시험: +x 24장 / −y 21장 모두 실제 깊이 0.05 mm 이내, 틈 높이 ±12 mm·회전 ±8°·아랫 모서리 시작·초점점이 출발점보다 바깥인 경우 모두 통과. 예상 실기 시간: 면당 약 10~12분.

## 실기 시도 기록 (모두 +x L13, Dino 손에 쥔 상태)
| 세션 | 결과 | 원인 → 조치 |
|---|---|---|
| `153548` | 3장 다 흐림(초점 바깥에서 멈춤) | 라플라시안 분산이 노이즈에 묻힘 → 가우시안 차이로 교체 |
| (붙여 준 출력) | +14 mm까지 초점 없음 | 그립이 바뀌어 면이 더 깊었음 → 탐색 한계 확대, 실패 시 다음 블록도 넓게 |
| `160722` | **+25 mm까지 들어가 L13 왼쪽 블록을 거의 밈** | 배율 50은 초점 ≈ 접촉, "최고점 지나 확인" 방식이 위험 → **다이얼 30으로**, 사람 결정 단계 추가 |
| `161033`, `162355`, `162906` | 사진 없음 | 사람 결정 모드인 줄 몰랐음 / 스페이스바가 화면 갱신 때 지워지는 버그 → 버그 수정, 자동 최고점 찾기로 변경 |
| `163402` | 출발점(−4 mm)이 최고(35), 들어갈수록 떨어지기만 함 → +13 mm까지 감 | 배율 30에서 초점점이 바깥으로 이동 → **떨어지기만 하면 바깥으로 탐색**(가상 시험 통과, 실기 미확인) |

## 다음에 할 일 (순서대로)
1. **배율 30 실기 재시험 (+x L13 한 층)**. Dino를 쥔 채 home에서:
   ```bash
   uv run jenga-autoscan scan --face px --layers 13 13 --seed cali_way_3 --via cali_way_1 \
       --send --line-speed 5 --prior-depth-mm 0 --prior-down-mm -8 --max-in-mm 25
   ```
   - 확인: `focus only falls … searching outwards`가 나오고 최고점을 찾는지, `top edge`가 맞는지(틈 위가 줄무늬), 수평·가운데 맞추기, 사진 3장의 틈.
   - 끝나면 `focus_log.csv`로 30배 초점 곡선의 폭을 보고 `--coarse-mm`(0.5) 조정. 30배 사진으로 `edge_side`의 줄무늬 차이도 다시 확인(50배 기준 값).
   - 초점이 잘 맞으면 **배율 30용으로 다시 보정**하는 게 좋다: 손으로 L01_S0을 맞추고 `jenga-calib`. 그러면 `--prior-depth-mm`이 0 근처로 돌아온다.
2. **30배에서 틈 px가 순위에 충분한지 판단**: 50배 기준 틈 중간값 6 px → 30배 약 3.6 px 예상. 부족하면 다이얼 40 등으로 절충(접근 안전과 해상도 사이).
3. **한 면 전체(+x, 24장)** → **−y 면(21장)**. −y는 home→면 이동에서 손목 j5가 약 180° 돈다(처음엔 지켜보기).
4. 두 세션으로 `jenga-autoscan rank` → `ranking.json`. 팀원 코드가 읽을 **경로와 형식**을 팀원과 맞춘다(지금: `{"ranking": [{"rank", "block": "jenga_L05_S1", "layer", "slot", "score_px", ...}]}`).
5. `jenga-task`에 스캔 단계 넣기: pick → autoscan(+x) → autoscan(−y) → place 를 한 번에.
6. 왼팔: 왼손 pick/place 티칭(오른손처럼 매뉴얼), `jenga-calib --arm left`, 면 −x, +y. 두 팔 결과를 합치면 블록 양 끝 평균(`mean`, 예전 최고 성능).
7. 팀원에게: **최신 URDF + 메시**(곧 받기로 함) → `jenga-sim`에 실제 로봇 모양, 손 장착 각도 확인. `jenga_tf`(`perception:=true`)로 실제 탑 위치를 읽어 `scan_setup.json`에 반영(읽기만, domain 0에서 `jenga_tf`를 직접 실행하지 않는다 — 블록 상태 초기화, `/tf_static` 발행).
8. 정리: `tools/fake_robot.py`는 `jenga-sim`으로 대체됨(삭제 검토). 하루 작업 최종 커밋(`jenga_world/`는 각자 .git이 있어 제외).

## 자주 쓰는 명령
```bash
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=0   # 실제 로봇 (77 = 가상)
ros2 daemon stop; ros2 topic hz /joint_states                 # 연결 확인
uv run jenga-task pick_dino --send --step --arm-speed 5       # Dino 꺼내기 (home에서 끝남)
uv run jenga-task place_dino --send --step --arm-speed 5      # Dino 꽂기 (home에서 시작, 탑에서 먼저 빼기)
uv run jenga-arm --arm right --send --speed 5                 # 조그 (frame dino), Save robot pose
uv run jenga-view --ends                                      # Dino 화면 (autoscan과 동시에 못 씀)
uv run jenga-calib <자세> --face px --layer 1 --slot 0 [--write]
uv run jenga-sim [--check TASK]                               # domain 77 가상 로봇 + RViz
uv run jenga-autoscan scan --face px ...                      # --send 없으면 가상 시험
uv run jenga-autoscan rank 세션1 세션2
```
