uint8 MAX_MOTOR_NUM = 9  # Matches the existing constant

# Control parameters
float64[9] kp        # Proportional gain
float64[9] kd        # Derivative gain
float64[9] p         # Position setpoint
float64[9] v         # Velocity setpoint
float64[9] t_ff      # Feedforward torque
float64[9] t_limit   # Torque limit
int8[9] break_relase # Brake release status
float64[9] kd_slave  # Slave derivative gain
float64[9] vel_slave  # Velocity setpoint for slave motors

# Motor IDs (variable length)
int32[] motor_id_list # List of motor IDs to control
