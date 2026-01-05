uint8 MAX_MOTOR_NUM = 9  # Kept for documentation purposes

# Position, velocity, and dynamics arrays
float64[9] p        # Position
float64[9] v        # Velocity
float64[9] c        # Current
float64[9] vol      # Voltage
float64[9] acc      # Acceleration

# CAN communication status
uint8[9] can_count
uint8[9] can_count_last

# For CanErrorStatus, you might need a separate message or define it here
# If using a simple approach for now:
uint8[9] can_error  # Error status for each motor

# Joint sensor data
bool[9] j_s         # Joint encoder status
float64[9] j_p      # Joint position (rad)
float64[9] j_v      # Joint velocity (rad/s)
float64[9] j_a      # Joint acceleration (rad/s^2)
