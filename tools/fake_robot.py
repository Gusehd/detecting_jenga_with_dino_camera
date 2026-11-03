"""Fake robot for testing the arm/hand tools WITHOUT the real robot.

Serves /compute_fk, /compute_ik (toy 7-DOF z-y-z-y-z-y-z arm, not the real
xArm geometry), the right-arm follow_joint_trajectory action, and /joint_states
for the right arm + right Allegro hand (hand commands are echoed back).

ALWAYS run it on a domain other than 0 (0 = real hardware), together with the tool:
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID=77
    python3 tools/fake_robot.py &
    uv run jenga-task pick_dino --send      # or jenga-arm / jenga-route / jenga-hand
"""
import os
if os.environ.get("ROS_DOMAIN_ID", "0") in ("", "0"):
    raise SystemExit("refusing to run on ROS_DOMAIN_ID 0 (real robot): export ROS_DOMAIN_ID=77")
import math, time, numpy as np, rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.srv import GetPositionFK, GetPositionIK
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from jenga_vision.cartesian import quat_to_matrix, matrix_to_quat
names = [f"xarm_right_joint{i}" for i in range(1, 8)]
AX = [2, 1, 2, 1, 2, 1, 2]; L = [0.267, 0, 0.293, 0, 0.3425, 0, 0.097]
def rot(a, t):
    c, s = math.cos(t), math.sin(t); i, j = (a+1)%3, (a+2)%3; r = np.eye(3); r[i,i]=c; r[i,j]=-s; r[j,i]=s; r[j,j]=c; return r
def fk(q):
    p = np.zeros(3); R = np.eye(3)
    for a, l, t in zip(AX, L, q):
        R = R @ rot(a, t); p = p + R @ np.array([0, 0, l])
    return p, R
def err(q, pg, Rg):
    p, R = fk(q); e = Rg @ R.T
    w = np.array([e[2,1]-e[1,2], e[0,2]-e[2,0], e[1,0]-e[0,1]]) / 2
    return np.concatenate([pg - p, w])
def ik(seed, pg, Rg):
    q = np.array(seed, float)
    for _ in range(200):
        e = err(q, pg, Rg)
        if np.linalg.norm(e) < 1e-7: return q
        J = np.zeros((6, 7))
        for k in range(7):
            dq = np.zeros(7); dq[k] = 1e-6; J[:, k] = (e - err(q + dq, pg, Rg)) / 1e-6
        q = q + J.T @ np.linalg.solve(J @ J.T + 1e-6*np.eye(6), e)
    return None
rclpy.init(); n = Node("fake_moveit"); cg = ReentrantCallbackGroup()
q = [0.0, 0.5, 0.0, 1.2, 0.0, 0.7, 0.0]
pub = n.create_publisher(JointState, "/joint_states", 10)
hnames = [f"ahr_joint{f}{j}" for f in range(5) for j in range(4)]
hq = [0.0]*20; hcount = [0]
from std_msgs.msg import Float64MultiArray
def on_hand(m):
    global hq
    hq = list(m.data); hcount[0] += 1
    if hcount[0] % 100 == 0: print(f"FAKE HAND: {hcount[0]} cmds, little j1 {math.degrees(hq[17]):.1f}, thumb j1 {math.degrees(hq[1]):.1f}", flush=True)
n.create_subscription(Float64MultiArray, "/allegro_hand_position_effort_controller_r/commands", on_hand, 10)
n.create_timer(0.05, lambda: pub.publish(JointState(name=names + hnames, position=q + hq)))
def fk_cb(req, res):
    js = req.robot_state.joint_state; qq = [js.position[js.name.index(x)] for x in names]
    p, R = fk(qq); ps = res.pose_stamped.__class__  # list
    from geometry_msgs.msg import PoseStamped
    m = PoseStamped(); m.header.frame_id = req.header.frame_id
    m.pose.position.x, m.pose.position.y, m.pose.position.z = p
    o = m.pose.orientation; o.x, o.y, o.z, o.w = matrix_to_quat(R)
    res.pose_stamped = [m]; res.fk_link_names = list(req.fk_link_names); res.error_code.val = 1; return res
def ik_cb(req, res):
    r = req.ik_request; js = r.robot_state.joint_state; seed = [js.position[js.name.index(x)] for x in names]
    pp = r.pose_stamped.pose; o = pp.orientation
    sol = ik(seed, np.array([pp.position.x, pp.position.y, pp.position.z]), quat_to_matrix(o.x, o.y, o.z, o.w))
    if sol is None: res.error_code.val = -31; return res
    res.solution.joint_state.name = names; res.solution.joint_state.position = [float(v) for v in sol]
    res.error_code.val = 1; return res
n.create_service(GetPositionFK, "/compute_fk", fk_cb, callback_group=cg)
n.create_service(GetPositionIK, "/compute_ik", ik_cb, callback_group=cg)
def exe(gh):
    global q
    pts = gh.request.trajectory.points
    print(f"FAKE CTRL: {len(pts)} points, T={pts[-1].time_from_start.sec + pts[-1].time_from_start.nanosec*1e-9:.2f}s, vel fields {[len(p.velocities) for p in pts][:3]}", flush=True)
    t0 = time.time(); prev_t = 0.0; q0 = list(q)
    for p in pts:
        T = p.time_from_start.sec + p.time_from_start.nanosec*1e-9
        while (time.time()-t0) < T:
            if gh.is_cancel_requested: print("FAKE CTRL: cancel received, j5 at", round(math.degrees(q[4]),1), flush=True); gh.canceled(); return FollowJointTrajectory.Result()
            s = ((time.time()-t0) - prev_t) / max(T-prev_t, 1e-6); q = [a+(b-a)*min(s,1) for a, b in zip(q0, p.positions)]; time.sleep(0.01)
        q = list(p.positions); q0 = list(q); prev_t = T
    gh.succeed(); return FollowJointTrajectory.Result()
ActionServer(n, FollowJointTrajectory, "/xarm7_right_controller/follow_joint_trajectory", exe,
             cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cg)
ex = MultiThreadedExecutor(); ex.add_node(n); ex.spin()
