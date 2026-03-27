"""Shared constants for Astribot S1 mesh processing pipeline."""

# Robot color scheme (RGB 0-1 float, for DAE diffuse materials)
WHITE_SHELL = (248 / 255, 249 / 255, 251 / 255)
BLACK_ACCENT = (20 / 255, 20 / 255, 22 / 255)

# Luminance threshold (uint8): below this → black accent
DARK_THRESHOLD = 50

# Logo zone parameters (torso_link_4 chest area)
# GLB coords: X=front, Y=up, Z=left-right
# OBJ Z-up coords: X=front, Y=-Z_glb, Z=Y_glb
LOGO_Z_CENTER = 0.38    # vertical center (Y_glb / Z_obj)
LOGO_Y_CENTER = 0.0     # horizontal center (Z_glb / -Y_obj)
ZONE_HALF = 0.055        # ±55mm zone around center
FRONT_THRESHOLD = 0.5    # face normal X component must exceed this

# S1 color palette (RGB uint8, for texture rasterization)
COLOR_WHITE_SHELL = (248, 249, 251)
COLOR_BLACK_VISOR = (20, 20, 22)
COLOR_DARK_MECHANISM = (55, 55, 60)
COLOR_CHASSIS_TOP = (246, 247, 250)
COLOR_CHASSIS_SIDE = (45, 45, 50)
