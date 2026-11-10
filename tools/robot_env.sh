# Real robot terminal (xArm7 + Allegro over the USB LAN, ROS_DOMAIN_ID 42 since 2026-11-09):
#   source tools/robot_env.sh
# Only for this terminal on purpose: sim terminals stay on 77 with the normal transports.
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=42
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_usb_lan.xml  # UDP only on 192.168.2.33, peer 192.168.2.32
ros2 daemon stop >/dev/null 2>&1
echo "robot terminal: ROS_DOMAIN_ID=$ROS_DOMAIN_ID, profile $FASTRTPS_DEFAULT_PROFILES_FILE"
