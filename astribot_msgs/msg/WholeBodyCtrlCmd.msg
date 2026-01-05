# Request
astribot_msgs/AstribotHeader header

# Twist for left and right arms (linear and angular velocity)
geometry_msgs/Twist left_arm_twist
geometry_msgs/Twist right_arm_twist
geometry_msgs/Twist torso_twist

# [x,y,z,qx,qy,qz,qw] desired poses in world frame for cartesian commands
float64[] pose_world_to_torso_desired
float64[] pose_world_to_left_arm_desired
float64[] pose_world_to_right_arm_desired
float64[] pose_world_to_chassis_current

bool torso_open_loop
bool enable_collision_avoidance

# # !!!It will not work!!! Reset WBC first loop flag if true
# bool first_flag

# Current smooth time (0.0 to smooth_duration)
float64 smooth_t
# Total duration for smooth transition (default: 1.0)
float64 smooth_duration
