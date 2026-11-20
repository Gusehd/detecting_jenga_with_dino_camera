# CES 현장 운영 순서 (2026-11-20)

Dino는 손 지그에 계속 꽂아 둔다고 가정한다. 명령은 동료 PC 기준(`jenga_autoscan` = `ros2 run jenga_vision jenga_autoscan`).

## 역할 나눔 (바꾸지 않음)
- **밑준비는 사람이 한다:** `mount`, `calib`. 상황에 따라 고르는 명령은 전부 여기 있다.
- **동료의 자동 반복 시퀀스에는 `run --send` 하나만 들어간다.** 분기가 없어야 하므로 run은 손가락을 고치거나
  보정하지 않고, 손가락이 보정과 8° 넘게 다르면 팔을 움직이지 않고 멈춘다(exit 1).
- `calib`과 `run`은 Enter를 묻지 않는다. Enter는 `mount`의 토크 단계에만 있다. 멈추기는 Ctrl+C.

## 1일차 아침 (운반 후 처음 켤 때): 전체 세팅
운반 중에 지그나 Dino가 밀렸을 수 있으므로 한 번은 처음부터 한다.
```bash
jenga_autoscan mount --send          # 토크 OFF → Dino가 끝까지 들어갔는지 손으로 확인 → Enter → 토크 ON → 오므림
jenga_autoscan calib --send --write  # 오늘 Dino 위치 재기 (3~4분), scan_setup.json에 저장
jenga_autoscan run --send            # 시험 한 판: times: 줄, WARNING 확인
```
- 탑은 평소 위치, 1층 끝면이 −y를 보게 쌓는다.
- 시험 판의 `times:`가 평소 수준이고 경고가 없으면 준비 끝.

## 데모 한 판마다 (자동 시퀀스)
```bash
jenga_autoscan run --send            # 동료 home에서 시작해 동료 home으로 끝, 순서는 game_block.json
```
- 탑을 새로 쌓아도 calib은 필요 없다(jig-full이 모든 층을 직접 찾음).

## 하루 중 손 토크만 풀렸을 때 (Dino와 지그는 그대로)
```bash
jenga_autoscan mount --send --no-torque   # 손가락만 마지막 calib 자세로 오므리고 확인 (팔 안 움직임, 스캔 안 함)
```
- `mounted`가 나오면 자동 시퀀스를 다시 돌린다.
- 돌아가는 자세는 `scan_setup.json`의 `dino.calibration.<arm>.hand_cmd`(없으면 `hand_q`): 마지막
  `calib --write` 때의 손가락 각도다.

## 하루 끝
- 팔은 동료 home(run이 끝나면 이미 거기 있음)에 두고 끈다.
- 전원이 꺼지면 손이 풀린다. Dino가 지그에서 빠지지 않는지만 보고 그대로 둔다.

## 2일차 이후 아침 (Dino를 건드리지 않았을 때)
```bash
jenga_autoscan mount --send --no-torque   # 어제 calib 자세로 오므림
jenga_autoscan run --send                 # 시험 한 판
```
시험 판이 평소대로면 calib 없이 그날 데모를 한다.

## 이게 보이면 1일차 아침 순서로 (mount --send → calib --send --write)
- `NOT mounted` 또는 `STOPPED: ring / little finger ...`
- run의 `WARNING ... focus searches may take longer`, 또는 `times:`가 평소보다 확 늘어남
- Dino가 손가락 사이에서 덜렁거림, Dino를 뺐거나 다시 꽂음

## 주의
- **동료 PC 빌드 전에 config 백업:** `colcon build`가 install share의 config를 덮어쓴다. pull/빌드 전에
  `scan_setup.json`, `hand_poses.json`을 백업하고 되돌려 넣는다. 현장 calib 값이 날아가면 처음부터 다시.
- Ctrl+C로 멈추면 팔은 그 자리에 선다. 탑 옆에서는 관절 이동으로 빼지 않는다.
- 아직 실기에서 안 해 본 것: 손 토크를 하루 종일 켜 두고 Dino를 쥐고 있기. 리허설 때 몇 시간 켜 두고 발열,
  토크 풀림을 확인한다.

## 다음 목표: 속도 (1순위)
−y, +x로 가기 전 대기 시간이 길다. 의심 순서:
1. 면마다 IK 계획 (캐시가 calib --write 뒤 무효; 이 PC 가상 실행에서 −y 6 s, +x 3 s, 예전 기록 −y 24 s).
   ROUTE로 처음·끝 관절이 이미 고정이라 생략하거나 미리 할 수 있다.
2. 면마다 ROS 노드를 새로 띄움 (rclpy init + 로봇 PC DDS 탐색, run 한 번에 4번).
3. 단계별 시간이 출력되지 않아 어디서 기다리는지 안 보임.
