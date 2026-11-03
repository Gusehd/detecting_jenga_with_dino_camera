# 로봇팔 제어 인터페이스 (sim_bi_manipulator_bringup)

`bi_manipulator_bringup/launch/sim_bi_manipulator_bringup.launch.py` 실행 시 사용 가능한 토픽 / 액션 / 서비스 정리입니다.
(실행 중인 시스템에서 `ros2 action list`, `ros2 topic list`, `ros2 service list`로 확인)

로봇팔을 움직이는 방법은 두 가지입니다.

1. **MoveIt2 사용** – 목표(포즈/관절값)를 주면 계획(충돌 회피 포함) + 실행
2. **컨트롤러에 직접 명령** – 관절 궤적을 직접 만들어 ros2_control 컨트롤러로 전송

---

## 1. MoveIt2 사용 (`move_group`)

### Action

| 이름 | 타입 | 용도 |
|---|---|---|
| `/move_action` | `moveit_msgs/action/MoveGroup` | 목표 → 계획 + 실행 (MoveGroupInterface가 내부적으로 사용) |
| `/execute_trajectory` | `moveit_msgs/action/ExecuteTrajectory` | 이미 계획된 궤적 실행 |
| `/sequence_move_group` | `moveit_msgs/action/MoveGroupSequence` | Pilz 다중 구간(PTP/LIN/CIRC) 연속 실행 |

### Service

| 이름 | 타입 | 용도 |
|---|---|---|
| `/plan_kinematic_path` | `moveit_msgs/srv/GetMotionPlan` | 계획만 수행 |
| `/plan_sequence_path` | `moveit_msgs/srv/GetMotionSequence` | Pilz 시퀀스 계획만 수행 |
| `/compute_cartesian_path` | `moveit_msgs/srv/GetCartesianPath` | 직선(Cartesian) 경로 계획 |
| `/compute_ik` | `moveit_msgs/srv/GetPositionIK` | IK 계산 |
| `/compute_fk` | `moveit_msgs/srv/GetPositionFK` | FK 계산 |

### 설정 정보

- **Planning group**
  - `xarm_right` : `xarm_right_link_base` → `ahr_base_link`
  - `xarm_left` : `xarm_left_link_base` → `ahl_base_link`
  - `dual_arm` : 양팔
- **Planner** : 기본 `pilz_industrial_motion_planner`, `ompl` 사용 가능
- MoveIt은 계획한 궤적을 아래 2번의 `follow_joint_trajectory` 액션으로 컨트롤러에 전달합니다.

---

## 2. 컨트롤러에 직접 명령 (JointTrajectoryController)

컨트롤러: `xarm7_right_controller`, `xarm7_left_controller` (`joint_trajectory_controller/JointTrajectoryController`, position 인터페이스)

| 이름 | 종류 | 타입 |
|---|---|---|
| `/xarm7_right_controller/follow_joint_trajectory` | Action | `control_msgs/action/FollowJointTrajectory` |
| `/xarm7_left_controller/follow_joint_trajectory` | Action | `control_msgs/action/FollowJointTrajectory` |
| `/xarm7_right_controller/joint_trajectory` | Topic | `trajectory_msgs/msg/JointTrajectory` |
| `/xarm7_left_controller/joint_trajectory` | Topic | `trajectory_msgs/msg/JointTrajectory` |

- **관절 이름** : `xarm_right_joint1` ~ `xarm_right_joint7`, `xarm_left_joint1` ~ `xarm_left_joint7`
- **Action** : 실행 완료/실패 결과를 받을 수 있음 (권장)
- **Topic** : 보내기만 하고 결과는 받지 않음
- ⚠️ 충돌 검사 없이 바로 움직이므로 실제 로봇에서는 주의

### 예시 (Topic)

```bash
ros2 topic pub --once /xarm7_right_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: [xarm_right_joint1, xarm_right_joint2, xarm_right_joint3, xarm_right_joint4, xarm_right_joint5, xarm_right_joint6, xarm_right_joint7],
  points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}"
```

### 예시 (Action)

```bash
ros2 action send_goal /xarm7_right_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory \
"{trajectory: {joint_names: [xarm_right_joint1, xarm_right_joint2, xarm_right_joint3, xarm_right_joint4, xarm_right_joint5, xarm_right_joint6, xarm_right_joint7],
  points: [{positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}}"
```

---

## 상태 확인용 토픽

| 이름 | 타입 |
|---|---|
| `/joint_states` | `sensor_msgs/msg/JointState` |
| `/xarm7_right_controller/controller_state` | `control_msgs/msg/JointTrajectoryControllerState` |
| `/xarm7_left_controller/controller_state` | `control_msgs/msg/JointTrajectoryControllerState` |
