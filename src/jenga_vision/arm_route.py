"""xArm7 routes: teach joint waypoints in manual mode, play them back slowly.

Needs ROS 2 Humble sourced and the right ROS_DOMAIN_ID. Playback sends each
waypoint straight to /xarm7_<arm>_controller/follow_joint_trajectory (no
MoveIt: no planning, no collision check), so by default nothing is sent (dry
run). Pass --send to move.

Between two waypoints the joints are interpolated, so the hand moves on an arc,
not a straight line. If a move passes too close to the tower, insert a via
point (v) between the two waypoints: via points are only passed through, shot
points are where the Dino will take a picture.

teach  (nothing is ever sent; move the arm by hand in manual mode)
  <Enter> [label]   add the arm's pose as a shot point
  v [label]         add it as a via point
  i N               insert the next points after point N (0 = at the start)
  r N               replace point N with the arm's pose
  d N               delete point N
  l                 list the route
  q                 quit (every change is saved right away)

play
  Moves through the points in order, each one so that no joint is faster than
  --speed deg/s (at least 2 s). By default it waits for Enter before every
  move (q quits); --auto runs through, pausing --dwell s at shot points.
  The first move goes from wherever the arm is to the first point.
  Ctrl+C cancels the move; the controller holds where the arm is.

Routes are saved in config/arm_routes.json (per arm, by name).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from jenga_vision.arm_slider import MIN_MOVE_TIME, ArmNode, clamp

ROUTES_FILE = Path("config/arm_routes.json")


def load_all() -> dict:
    try:
        return json.loads(ROUTES_FILE.read_text())
    except (OSError, ValueError):
        return {}


def load_route(arm: str, name: str) -> list[dict]:
    return load_all().get(arm, {}).get(name, [])


def save_route(arm: str, name: str, points: list[dict]) -> None:
    data = load_all()
    data.setdefault(arm, {})[name] = points
    ROUTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    ROUTES_FILE.write_text(json.dumps(data, indent=2))


def deg(values: list[float]) -> str:
    return " ".join(f"{math.degrees(v):7.1f}" for v in values)


def biggest_step(a: list[float], b: list[float]) -> float:
    """Largest joint change between two poses (deg)."""
    return max(abs(math.degrees(x - y)) for x, y in zip(a, b))


def move_time(a: list[float], b: list[float], speed: float) -> float:
    return max(MIN_MOVE_TIME, biggest_step(a, b) / max(speed, 0.1))


def describe(n: int, p: dict) -> str:
    kind = "via " if p["via"] else "shot"
    return f"{n:3d} {kind} {p['label'] or '-':<16} {deg(p['q'])}"


def print_route(points: list[dict]) -> None:
    if not points:
        print("  (empty)")
    print("    #  kind label            " + " ".join(f"{'j' + str(i + 1):>7}" for i in range(7)))
    for n, p in enumerate(points, 1):
        step = f"  (+{biggest_step(p['q'], points[n - 2]['q']):.1f} deg)" if n > 1 else ""
        print("  " + describe(n, p) + step)


def wait_pose(ros: ArmNode) -> list[float]:
    for _ in range(50):
        cur = ros.current()
        if cur is not None:
            return cur
        time.sleep(0.1)
    raise SystemExit(f"no /joint_states for {ros.names[0]}..7 (ROS sourced? ROS_DOMAIN_ID?)")


# --- teach -------------------------------------------------------------------------------
def teach(ros: ArmNode, args: argparse.Namespace) -> None:
    points = load_route(args.arm, args.route)
    insert_at = len(points)  # index in points where the next new point goes
    print(f"route '{args.route}' ({args.arm} arm), {len(points)} points. Nothing is sent to the arm.")
    print_route(points)
    while True:
        cur = wait_pose(ros)
        where = "end" if insert_at == len(points) else f"after point {insert_at}"
        line = input(f"\n[{len(points)} pts, new -> {where}]  arm {deg(cur)}\n"
                     "Enter=shot  v=via  i N  r N  d N  l  q > ").strip()
        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        cmd_l = cmd.lower()
        if cmd_l == "q":
            break
        if cmd_l == "l":
            print_route(points)
            continue
        if cmd_l in ("i", "r", "d"):
            try:
                n = int(rest)
            except ValueError:
                print("needs a point number, e.g. 'i 2'")
                continue
            if cmd_l == "i":
                if 0 <= n <= len(points):
                    insert_at = n
                else:
                    print(f"point number 0..{len(points)}")
                continue
            if not 1 <= n <= len(points):
                print(f"point number 1..{len(points)}")
                continue
            if cmd_l == "r":
                points[n - 1]["q"] = [round(v, 5) for v in ros.current()]
                print("replaced " + describe(n, points[n - 1]))
            else:
                print("deleted " + describe(n, points.pop(n - 1)))
                if insert_at >= n:
                    insert_at -= 1
            save_route(args.arm, args.route, points)
            continue
        if cmd_l == "v":
            via, label = True, rest
        elif cmd_l in ("", "s"):
            via, label = False, rest
        else:  # anything else is the label of a shot point
            via, label = False, line
        q = [round(v, 5) for v in ros.current()]
        points.insert(insert_at, {"label": label, "via": via, "q": q})
        insert_at += 1
        save_route(args.arm, args.route, points)
        prev = f"  (+{biggest_step(q, points[insert_at - 2]['q']):.1f} deg from previous)" if insert_at > 1 else ""
        print("added " + describe(insert_at, points[insert_at - 1]) + prev)
    print(f"saved {len(points)} points to {ROUTES_FILE}")


# --- play --------------------------------------------------------------------------------
def run_move(ros: ArmNode, target: list[float], seconds: float) -> bool:
    """Send one move and block until it finishes. Ctrl+C cancels it."""
    ros.move(target, seconds)
    try:
        while not ros.done.wait(0.1):
            pass
    except KeyboardInterrupt:
        ros.stop()
        print("\nCtrl+C: move cancelled, arm holds")
        ros.done.wait(3)
        raise
    print(f"      {ros.result}")
    return ros.succeeded


def play(ros: ArmNode, args: argparse.Namespace) -> None:
    points = load_route(args.arm, args.route)
    if not points:
        raise SystemExit(f"no route '{args.route}' for the {args.arm} arm in {ROUTES_FILE}")
    last = len(points) if args.to is None else args.to
    todo = list(range(args.start, last + 1))
    if not todo or not 1 <= args.start <= last <= len(points):
        raise SystemExit(f"--from/--to must be within 1..{len(points)}")
    for n in todo:
        q = points[n - 1]["q"]
        if any(abs(clamp(i, v) - v) > 1e-6 for i, v in enumerate(q)):
            raise SystemExit(f"point {n} is outside the joint limits")
    if args.send and not ros.client.wait_for_server(timeout_sec=5):
        raise SystemExit(f"action server not available: {ros.action_name}")

    mode = "SEND" if args.send else "DRY RUN (nothing is sent)"
    pace = "auto" if args.auto else "Enter before every move"
    print(f"route '{args.route}' ({args.arm} arm) points {todo[0]}-{todo[-1]}, {args.speed} deg/s, {pace}, {mode}")
    cur = wait_pose(ros)
    for n in todo:
        p = points[n - 1]
        start = cur if args.send or n == todo[0] else points[n - 2]["q"]
        seconds = move_time(start, p["q"], args.speed)
        print(f"\n-> {describe(n, p)}\n   largest joint change {biggest_step(start, p['q']):.1f} deg, {seconds:.1f} s")
        if n == todo[0]:
            print("   (first move: from where the arm is now)")
        if not args.auto:
            if input("   Enter = go, q = quit > ").strip().lower() == "q":
                break
        if not args.send:
            continue
        if not run_move(ros, p["q"], seconds):
            print("   move did not succeed: stopping")
            break
        cur = wait_pose(ros)
        err = biggest_step(cur, p["q"])
        print(f"   reached, largest joint error {err:.2f} deg")
        if args.auto and not p["via"] and args.dwell > 0:
            time.sleep(args.dwell)
    print("\nend of playback")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_teach = sub.add_parser("teach", help="record waypoints (moves nothing)")
    p_play = sub.add_parser("play", help="move through the waypoints")
    p_list = sub.add_parser("list", help="print routes")
    for p in (p_teach, p_play, p_list):
        p.add_argument("--arm", choices=("left", "right"), required=True)
    p_teach.add_argument("route", help="route name, e.g. f1")
    p_play.add_argument("route")
    p_list.add_argument("route", nargs="?")
    p_play.add_argument("--send", action="store_true", help="really move the arm (default: dry run)")
    p_play.add_argument("--speed", type=float, default=5.0, help="max joint speed (deg/s), default 5")
    p_play.add_argument("--from", dest="start", type=int, default=1, help="first point (1-based)")
    p_play.add_argument("--to", type=int, default=None, help="last point")
    p_play.add_argument("--auto", action="store_true", help="don't wait for Enter before each move")
    p_play.add_argument("--dwell", type=float, default=1.0, help="--auto: pause at shot points (s)")
    args = parser.parse_args()

    if args.cmd == "list":
        routes = load_all().get(args.arm, {})
        for name in ([args.route] if args.route else sorted(routes)):
            print(f"{name}:")
            print_route(routes.get(name, []))
        return

    try:
        ros = ArmNode(args.arm, args.cmd == "play" and args.send, node_name=f"jenga_route_{args.cmd}")
    except ImportError:
        raise SystemExit("rclpy not found: run `source /opt/ros/humble/setup.bash` first")
    try:
        (teach if args.cmd == "teach" else play)(ros, args)
    except (KeyboardInterrupt, EOFError):
        print("\nquit")
    finally:
        ros.shutdown()


if __name__ == "__main__":
    main()
