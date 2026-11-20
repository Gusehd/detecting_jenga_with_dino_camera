# CES 현장 운영 순서 (2026-11-20)

Dino는 손 지그에 계속 꽂아 둔다고 가정한다. 명령은 동료 PC 기준(`jenga_autoscan` = `ros2 run jenga_vision jenga_autoscan`).
2026-11-20 동료 PC에서 teach → calib → `mount --no-torque`(0.7° 안으로 돌아옴)까지 확인했다.

## 역할 나눔 (바꾸지 않음)
- **밑준비는 사람이 한다:** `mount`, `calib`. 상황에 따라 고르는 명령은 전부 여기 있다.
- **동료의 자동 반복 시퀀스에는 `run --send` 하나만 들어간다.** 분기가 없어야 하므로 run은 손가락을 고치거나
  보정하지 않고, 손가락이 보정과 8° 넘게 다르면 팔을 움직이지 않고 멈춘다(exit 1).
- `calib`과 `run`은 Enter를 묻지 않는다(`8aa3d34`부터). Enter는 `mount`의 토크 단계에만 있다. 멈추기는 Ctrl+C.
- 손가락 1°에 Dino가 약 2 mm 움직인다(11-20 calib 예측). 손 검사(8°)를 빼면 안 되는 이유.

## 0일차: 지그를 손에 처음 달았을 때, Dino를 새로 꽂았을 때 (한 번)
팔은 `jig_ready_1118`(탑에서 떨어진 자세)에 둔다. calib이 거기서 출발한다.
```bash
jenga_autoscan mount --teach --send
#   토크 OFF → Dino를 지그 끝까지, 축 방향은 지난번과 같게 → 약지·새끼를 손으로 Dino에 닿게 → Enter → 토크 ON
#   → hand_poses.json에 jig_mount_<날짜> 저장. "fingers moved X deg"가 나오면 손이 못 버틴 것
jenga_autoscan calib --send --hand jig_mount_<날짜> --write
#   "saved to .../scan_setup.json"이 나와야 함. nothing saved면 보정은 예전 그대로(아래 mount도 예전 자세로 감)
jenga_autoscan mount --send --no-torque
#   "mounted: ... within 1~2 deg"면 이 손 모양을 다시 만들 수 있다는 확인
```
그다음 탑을 쌓고(평소 위치, 1층 끝면이 −y) 시험 한 판:
```bash
jenga_autoscan run --send            # times: 줄, WARNING 확인
```

## 매일 아침 (지그와 Dino를 안 건드렸을 때)
```bash
jenga_autoscan mount --send --no-torque   # 마지막 calib 자세로 오므림 + 확인 (팔 안 움직임, 스캔 안 함)
jenga_autoscan run --send                 # 시험 한 판
```
- calib은 다시 안 한다. mount 끝의 `Now: jenga_autoscan calib --send --write, then run --send`는 늘 나오는
  안내라 이 경우엔 무시.
- 시험 판이 평소대로면 그날 데모.

## 데모 한 판마다 (자동 시퀀스)
```bash
jenga_autoscan run --send            # 동료 home → −y → +x → 동료 home, 순서는 game_block.json
```
- 탑을 새로 쌓아도 calib은 필요 없다(jig-full이 모든 층을 직접 찾음).

## 하루 중 손 토크만 풀렸을 때
```bash
jenga_autoscan mount --send --no-torque   # mounted면 자동 시퀀스 재개
```

## 이게 보이면 calib부터 다시
- `NOT mounted`, 또는 run의 `STOPPED: ring / little finger ...`
- run의 `WARNING ... focus searches may take longer`, `times:`가 평소보다 확 늘어남
- Dino가 덜렁거림, Dino를 뺐거나 다시 꽂음

같은 손 모양으로 다시 꽂을 때 (teach한 자세는 그대로 씀):
```bash
jenga_autoscan mount --send            # 토크 OFF → Dino 꽂기 → Enter → 토크 ON → 마지막 calib 자세로 오므림
jenga_autoscan calib --send --write    # 팔은 jig_ready_1118에서
jenga_autoscan run --send
```
mount가 이것도 `NOT mounted`면(손가락이 Dino에 먼저 막힘) 0일차 순서(teach부터).

## 하루 끝
- 팔은 동료 home(run이 끝나면 이미 거기)에 두고 끈다.
- 전원이 꺼지면 손이 풀린다. Dino가 지그에서 빠지지 않는지만 보고 그대로 둔다.

## 주의
- **설정 파일 위치:** 11-20 동료 PC는 체크아웃의 `config/`를 쓴다(session이 `src/.../captures`에 생김). pull이나
  빌드 전에 `config/`를 백업한다. install share를 쓰는 빌드라면 `colcon build`가 덮어쓰니 거기도 백업.
- calib 전에 팔이 `jig_ready_1118`에 있는지 본다(11-20에 다른 자세에서 출발해 161° 관절 이동이 나왔음).
- Ctrl+C로 멈추면 팔은 그 자리에 선다. 탑 옆에서는 관절 이동으로 빼지 않는다.
- 아직 실기에서 안 해 본 것: 손 토크를 하루 종일 켜 두기. 리허설 때 몇 시간 켜 두고 `jenga_autoscan hand`로
  각도가 그대로인지 본다.
- calib 알려진 문제: 사진 위아래를 뒤집은 직후 모서리를 다시 안 찾아 `no edge to measure`로 멈출 수 있다
  (11-20). Dino를 지난번과 같은 축 방향으로 꽂으면 피할 가능성이 크다.

## 다음 목표: 속도 (1순위)
−y, +x로 가기 전 대기 시간이 길다(11-19 눈으로 약 15 s). 의심 순서:
1. 면마다 IK 계획 (캐시가 calib --write 뒤 무효; 이 PC 가상 실행에서 −y 6 s, +x 3 s, 예전 기록 −y 24 s).
   ROUTE로 처음·끝 관절이 이미 고정이라 하드코딩한다.
2. 면마다 ROS 노드를 새로 띄움 (rclpy init + 로봇 PC DDS 탐색, run 한 번에 4번). 노드 하나로 run 내내.
3. 단계별 시간이 출력되지 않아 어디서 기다리는지 안 보임.
