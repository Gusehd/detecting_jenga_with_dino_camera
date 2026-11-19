# Notes for AI coding agents

Read `README.md` first ("통합할 때 알아야 할 점"), then the top section of `docs/progress_2026-11-02_robot.md`.

## Safety (real robot)
- ROS_DOMAIN_ID 42 (and 0) drives the real bimanual xArm7 + Allegro hands. Never publish, call a service/action, or run a command with `--send` on those domains yourself. Hand the exact command to the human, who runs it while watching the robot. Read-only inspection (`ros2 topic list/info/echo`) is fine.
- Domain 77 is the simulation (`uv run jenga-sim`). Test there first, with `--fake` for the camera.
- Tools that can move the robot must stay dry-run by default and need an explicit `--send`.
- Moves go straight to `/xarm7_<arm>_controller/follow_joint_trajectory` with no collision checking. Never start a joint move (pick, place, `--via`) from a pose next to the tower.

## Environment
- Python only through the uv project: `~/.local/bin/uv run ...`, dependencies with `uv add`. No system pip. ROS 2 Humble comes from `source /opt/ros/humble/setup.bash` (or `source tools/robot_env.sh` for the real robot).
- The package is meant to be ported into the robot system as a module: keep the `src/jenga_vision` library layout and CLI entry points in `pyproject.toml`.
- `ros2/jenga_vision/` is the ROS 2 (ament_python, Humble and Jazzy) build of the same code through symlinks; its `setup.py` reads the scripts and version from `pyproject.toml` (`-` becomes `_`). Don't move it to the repo root. Paths to `config/`, `captures/` and the URDF go through `jenga_vision.paths`, never `Path("config/...")`.

## Contract with the pulling code
- Output: `captures/order_*.json`, copied to `$JENGA_STATE_DIR/jenga_pick_candidates.json` when that variable is set. Format and meaning (block names `jenga_LXX_SY`, L00 = bottom, S1 = middle; list order = pull order; 빼안빼 pairs) are in the README. Changing the format breaks `jenga_game`: agree on it with both owners first.

## Things that look wrong but are deliberate
- `config/arm_poses.json` poses were taught by hand on the robot. Don't regenerate or "clean up" them. Back up before any change.
- The Dino dial stays at 40 during a scan. Focus is found by moving the arm, not the lens.
- The Dino shifts in the hand on every pick (depth 2-3 mm, roll about 5 deg). Never trust a fixed depth across picks: that is why each face has reference searches.
- The tower must be stacked with floor-1 end faces toward -y.
