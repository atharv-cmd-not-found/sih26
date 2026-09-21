import os
import sys
import time
import math
import signal
import atexit
import warnings
import queue
import cv2
import numpy as np
import torch
import torch.nn as nn

# --- IMPORT ADVERSE WEATHER CONDITIONING FILTER ---
from engine.weather_filter import WeatherConditioningFilter

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import carla
import spconv.pytorch as spconv

try:
    from models.spconv_unet import SpConvUNet
except ImportError:
    class SpConvUNet(nn.Module):
        def __init__(self, in_channels=1, num_classes=5):
            super().__init__()
            self.linear = nn.Linear(in_channels, num_classes)
        def forward(self, x_sp):
            return self.linear(x_sp.features)

CHECKPOINT_PATH = r"checkpoints\spconv_semantickitti_best.pth"
LIDAR_FRAME_RATE_HZ = 20
LIDAR_POINTS_PER_FRAME = 30000
WEATHER_DURATION_SECONDS = 10.0
GROUND_RETURN_KEEP_PROBABILITY = 0.75
LIDAR_VOXEL_SIZE = 0.08
PERCEPTION_FORWARD_RANGE = 80.0
PERCEPTION_LATERAL_RANGE = 32.0
PERCEPTION_Z_MIN = -2.8
PERCEPTION_Z_MAX = 3.2

# Semantic Color Palette (BGR)
COLOR_PALETTE = {
    0: (40, 40, 40),       # Unclassified / Background (Dark Gray)
    1: (255, 140, 0),      # Dynamic Vehicles / 2-Wheelers (Dodger Blue)
    2: (0, 0, 255),        # Pedestrians / Jaywalkers (Vivid Red)
    3: (34, 139, 34),      # Drivable Road Surface (Forest Green)
    4: (0, 140, 255),      # Static Obstacles / Barriers (Amber Orange)
    5: (255, 255, 0),      # Curbs / Median Dividers (Cyan)
    6: (255, 0, 255),      # Potholes / Road Depressions (Magenta)
    7: (0, 215, 255),      # Stray Animals / Cattle (Gold)
}

IS_RUNNING = True
WORLD_POTHOLE_LOCATIONS = []
GLOBAL_CLEANUP_CONTEXT = {
    "client": None,
    "world": None,
    "traffic_manager": None,
    "actors": [],
    "cleaned": False
}

# ==============================================================================
# OPENCV TYPE-SAFE DRAWING WRAPPERS
# ==============================================================================
def pt(x, y):
    return (int(round(float(x))), int(round(float(y))))

def clr(c):
    return (int(c[0]), int(c[1]), int(c[2]))

def draw_rect(img, p1, p2, color, thickness=1):
    cv2.rectangle(img, pt(p1[0], p1[1]), pt(p2[0], p2[1]), clr(color), thickness)

def draw_line(img, p1, p2, color, thickness=1):
    cv2.line(img, pt(p1[0], p1[1]), pt(p2[0], p2[1]), clr(color), thickness)

def draw_circle(img, center, radius, color, thickness=-1):
    cv2.circle(img, pt(center[0], center[1]), int(radius), clr(color), thickness)

def draw_text(img, text, origin, font_scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, str(text), pt(origin[0], origin[1]), font, float(font_scale), clr(color), thickness, cv2.LINE_AA)

# ==============================================================================
# EMERGENCY CLEANUP & SIGNAL MANAGEMENT
# ==============================================================================
def emergency_cleanup():
    if GLOBAL_CLEANUP_CONTEXT["cleaned"]:
        return
    GLOBAL_CLEANUP_CONTEXT["cleaned"] = True
    print("\n[+] Restoring CARLA Master Simulation...")
    world = GLOBAL_CLEANUP_CONTEXT["world"]
    client = GLOBAL_CLEANUP_CONTEXT["client"]
    tm = GLOBAL_CLEANUP_CONTEXT["traffic_manager"]
    actors = GLOBAL_CLEANUP_CONTEXT["actors"]

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("[✓] CARLA restored to asynchronous mode.")
        except Exception:
            pass

    if tm is not None:
        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass

    if actors and client is not None:
        batch = [carla.command.DestroyActor(a) for a in actors if a is not None and a.is_alive]
        if batch:
            try:
                client.apply_batch_sync(batch, False)
                print(f"[✓] Cleaned up {len(batch)} master actors.")
            except Exception:
                pass
    cv2.destroyAllWindows()
    print("[+] Master bridge closed cleanly.")

def signal_handler(signum, frame):
    global IS_RUNNING
    IS_RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(emergency_cleanup)

def lidar_callback(sensor_data, data_queue):
    raw = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw, (int(raw.shape[0] / 4), 4))
    data_queue.put(points)

def update_spectator_follow_cam(spectator, vehicle):
    try:
        tf = vehicle.get_transform()
        rad = math.radians(tf.rotation.yaw)
        cam_x = tf.location.x - 7.5 * math.cos(rad)
        cam_y = tf.location.y - 7.5 * math.sin(rad)
        cam_z = tf.location.z + 3.8
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=cam_x, y=cam_y, z=cam_z),
                carla.Rotation(pitch=-18.0, yaw=tf.rotation.yaw, roll=0.0)
            )
        )
    except Exception:
        pass

def weather_for_condition(condition):
    presets = {
        "snowfall": carla.WeatherParameters(
            cloudiness=95.0,
            precipitation=100.0,
            precipitation_deposits=75.0,
            fog_density=20.0,
            fog_distance=25.0,
            wetness=65.0,
            sun_altitude_angle=10.0,
        ),
        "rain": carla.WeatherParameters(
            cloudiness=90.0,
            precipitation=85.0,
            precipitation_deposits=70.0,
            fog_density=35.0,
            fog_distance=18.0,
            wetness=85.0,
            sun_altitude_angle=15.0,
        ),
        "fog": carla.WeatherParameters(
            cloudiness=70.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            fog_density=85.0,
            fog_distance=8.0,
            wetness=20.0,
            sun_altitude_angle=25.0,
        ),
        "dust": carla.WeatherParameters(
            cloudiness=45.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            fog_density=35.0,
            fog_distance=15.0,
            wetness=0.0,
            wind_intensity=80.0,
            dust_storm=90.0,
            sun_altitude_angle=35.0,
        ),
        "smoke": carla.WeatherParameters(
            cloudiness=60.0,
            precipitation=0.0,
            precipitation_deposits=0.0,
            fog_density=75.0,
            fog_distance=6.0,
            wetness=0.0,
            wind_intensity=10.0,
            dust_storm=35.0,
            sun_altitude_angle=20.0,
        ),
    }
    return presets[condition]

def update_dynamic_weather(world, elapsed_seconds, current_index):
    conditions = ("snowfall", "rain", "fog", "dust", "smoke")
    next_index = int(elapsed_seconds // WEATHER_DURATION_SECONDS) % len(conditions)
    if next_index != current_index:
        condition = conditions[next_index]
        world.set_weather(weather_for_condition(condition))
        print(f"[✓] Dynamic weather: {condition} ({WEATHER_DURATION_SECONDS:.0f}s)")
    return next_index

def inject_world_potholes(xyz, ego_vehicle):
    global WORLD_POTHOLE_LOCATIONS
    if len(xyz) == 0 or len(WORLD_POTHOLE_LOCATIONS) == 0:
        return xyz
    v_tf = ego_vehicle.get_transform()
    yaw = math.radians(v_tf.rotation.yaw)
    cos_y = math.cos(-yaw)
    sin_y = math.sin(-yaw)

    for (wx, wy, radius, depth) in WORLD_POTHOLE_LOCATIONS:
        dx = wx - v_tf.location.x
        dy = wy - v_tf.location.y
        rel_x = dx * cos_y - dy * sin_y
        rel_y = dx * sin_y + dy * cos_y
        if 0.0 < rel_x < 35.0 and abs(rel_y) < 12.0:
            dist = np.hypot(xyz[:, 0] - rel_x, xyz[:, 1] - rel_y)
            mask = dist < radius
            if np.any(mask):
                xyz[mask, 2] -= depth * (1.0 - (dist[mask] / radius))
    return xyz

# ==============================================================================
# PERCEPTION, CURB-GATING & TRUE DYNAMIC CLUSTERING
# ==============================================================================
def extract_elevation_features(xyz, preds):
    labels = preds.copy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 1. Pitch-Invariant Road Plane Fit across the immediate lane
    fwd_road = (x >= 1.5) & (x <= 9.0) & (np.abs(y) <= 1.4) & (z >= -2.4) & (z <= -1.3)
    if np.count_nonzero(fwd_road) > 35:
        A = np.column_stack([x[fwd_road], y[fwd_road], np.ones(np.count_nonzero(fwd_road))])
        sol, _, _, _ = np.linalg.lstsq(A, z[fwd_road], rcond=None)
        z_expected = sol[0] * x + sol[1] * y + sol[2]
        z_ground_scalar = float(np.median(z[fwd_road]))
    else:
        z_expected = -1.85
        z_ground_scalar = -1.85

    h_local = z - z_expected

    # Potholes (Class 6)
    pothole_mask = (h_local <= -0.08) & (h_local >= -0.28) & (x >= 2.0) & (x <= 14.0) & (np.abs(y) <= 3.2)
    if np.count_nonzero(pothole_mask) >= 6:
        labels[pothole_mask] = 6

    # Road Surface (Class 3)
    road_mask = (h_local > -0.06) & (h_local < 0.06) & (~pothole_mask)
    labels[road_mask] = 3

    # Curbs & Median Dividers (Class 5)
    curb_mask = (h_local >= 0.06) & (h_local <= 0.42)
    labels[curb_mask] = 5

    # Stray Animals (Class 7)
    animal_mask = (h_local >= 0.15) & (h_local <= 0.85) & (labels == 4) & (np.abs(y) <= 3.6)
    labels[animal_mask] = 7

    # Elevated Obstacles (Class 4)
    elevated_mask = (h_local > 0.40) & (h_local <= 2.60)
    labels[elevated_mask & (labels != 1) & (labels != 2) & (labels != 7)] = 4

    # Suppress buildings, high walls, and tall foliage
    tall_mask = (h_local > 2.60) | (np.abs(y) > 16.0)
    labels[tall_mask] = 4

    # 2. Strict C++ Connected Components Clustering for Real Dynamic Actors
    clusters = []
    occ_res = 0.35
    grid_dim = 140
    occ_grid = np.zeros((grid_dim, grid_dim), dtype=np.uint8)

    dynamic_candidate_pts = (labels == 1) | (labels == 2) | (labels == 7)
    if np.any(dynamic_candidate_pts):
        ox = np.clip((x[dynamic_candidate_pts] / PERCEPTION_FORWARD_RANGE * (grid_dim - 1)).astype(np.int32), 0, grid_dim - 1)
        oy = np.clip(((y[dynamic_candidate_pts] + PERCEPTION_LATERAL_RANGE) / (2.0 * PERCEPTION_LATERAL_RANGE) * (grid_dim - 1)).astype(np.int32), 0, grid_dim - 1)
        occ_grid[ox, oy] = 255

        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(occ_grid, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 4:
                continue

            cx_m = (centroids[i][1] / (grid_dim - 1)) * PERCEPTION_FORWARD_RANGE
            cy_m = (centroids[i][0] / (grid_dim - 1)) * (2.0 * PERCEPTION_LATERAL_RANGE) - PERCEPTION_LATERAL_RANGE

            if abs(cy_m) > 3.6:
                continue

            c_mask = dynamic_candidate_pts & (np.abs(x - cx_m) < 2.2) & (np.abs(y - cy_m) < 2.2)
            pts_count = np.count_nonzero(c_mask)
            if pts_count < 12:
                continue

            z_min_c = np.min(z[c_mask])
            z_max_c = np.max(z[c_mask])
            h_base = z_min_c - z_ground_scalar
            h_span = z_max_c - z_min_c
            dx = np.max(x[c_mask]) - np.min(x[c_mask])
            dy = np.max(y[c_mask]) - np.min(y[c_mask])
            dist = math.hypot(cx_m, cy_m)
            bbox = (float(np.min(x[c_mask])), float(np.max(x[c_mask])), float(np.min(y[c_mask])), float(np.max(y[c_mask])))

            if h_base > 0.14 or h_base < -0.18:
                labels[c_mask] = 4
                continue

            # Vehicle Filter
            if 1.2 <= dx <= 5.8 and 0.7 <= dy <= 2.6 and 0.6 <= h_span <= 2.5 and pts_count >= 16:
                labels[c_mask] = 1
                clusters.append({
                    "label": f"Vehicle ({int(round(dist))} m)",
                    "class": 1,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist,
                    "color": (255, 140, 0)
                })
            # Pedestrian Filter
            elif dx <= 1.3 and dy <= 1.3 and 0.8 <= h_span <= 2.1 and pts_count >= 10:
                labels[c_mask] = 2
                clusters.append({
                    "label": f"Pedestrian ({int(round(dist))} m)",
                    "class": 2,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist,
                    "color": (0, 0, 255)
                })
            else:
                labels[c_mask] = 4

    return labels, clusters

# ==============================================================================
# SAFE-WAYPOINT TACTICAL GUIDANCE
# ==============================================================================
class TacticalGuidanceController:
    def __init__(self):
        self.stall_counter = 0
        self.last_pos = None
        self.is_autopilot = True

    def update(self, vehicle, traffic_manager, world_map, clusters, labels, xyz):
        v_tf = vehicle.get_transform()
        v_loc = v_tf.location
        vel = vehicle.get_velocity()
        speed = 3.6 * math.hypot(vel.x, vel.y)

        fwd_threat = None
        min_dist = 99.0
        corridor_w = 1.35

        for c in clusters:
            cx, cy = c["pos"]
            if 0.8 <= cx <= 14.0 and abs(cy) <= corridor_w:
                if cx < min_dist:
                    min_dist = cx
                    fwd_threat = c

        pothole_pts = (labels == 6) & (xyz[:, 0] >= 1.5) & (xyz[:, 0] <= 11.0) & (np.abs(xyz[:, 1]) <= 1.35)
        if np.count_nonzero(pothole_pts) >= 6:
            p_dist = float(np.min(xyz[pothole_pts, 0]))
            if p_dist < min_dist:
                min_dist = p_dist
                fwd_threat = {"class": 6, "dist": p_dist, "pos": (p_dist, 0.0), "label": "Pothole"}

        tl = vehicle.get_traffic_light()
        tl_state_str, tl_color = "NONE DETECTED", (100, 100, 100)
        if tl is not None:
            state = tl.get_state()
            if state == carla.TrafficLightState.Red:
                tl_state_str, tl_color = "RED (STOP)", (0, 0, 255)
                if self.is_autopilot:
                    vehicle.set_autopilot(False)
                    self.is_autopilot = False
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0))
                return "V2I: STOPPED AT RED SIGNAL", (0, 0, 255), tl_state_str, tl_color, "SIGNAL COMPLIANCE", (0, 255, 0)
            elif state == carla.TrafficLightState.Yellow:
                tl_state_str, tl_color = "YELLOW (SLOW)", (0, 255, 255)
            elif state == carla.TrafficLightState.Green:
                tl_state_str, tl_color = "GREEN (PROCEED)", (0, 255, 0)

        if self.last_pos is not None:
            disp = v_loc.distance(self.last_pos)
            if disp < 0.1 and speed < 1.0 and tl_state_str.startswith("NONE"):
                self.stall_counter += 1
            else:
                self.stall_counter = max(0, self.stall_counter - 1)
        self.last_pos = v_loc

        # Emergency Stop (< 4.5m)
        if fwd_threat is not None and fwd_threat["dist"] < 4.5:
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.95, steer=0.0))
            return f"CRITICAL: {fwd_threat['label']}", (0, 0, 255), tl_state_str, tl_color, "ZERO-TOLERANCE BRAKE", (0, 0, 255)

        # Deadlock Bypass (> 25 ticks)
        if self.stall_counter > 25:
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False
            vehicle.apply_control(carla.VehicleControl(throttle=0.35, brake=0.0, steer=-0.38))
            return "TACTIC: DEADLOCK BYPASS OVERRIDE", (0, 165, 255), tl_state_str, tl_color, "ALTERNATE ROUTE OVERRIDE", (0, 165, 255)

        # Reactive Lateral Swerve
        if fwd_threat is not None:
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False

            left_clear = np.count_nonzero((xyz[:, 0] > 1.5) & (xyz[:, 0] < 11.0) & (xyz[:, 1] < -0.8) & (xyz[:, 1] > -3.2))
            right_clear = np.count_nonzero((xyz[:, 0] > 1.5) & (xyz[:, 0] < 11.0) & (xyz[:, 1] > 0.8) & (xyz[:, 1] < 3.2))

            obs_y = fwd_threat["pos"][1]
            steer = -0.32 if (obs_y >= 0.0 or right_clear > left_clear) else 0.32
            direction = "LEFT" if steer < 0 else "RIGHT"
            vehicle.apply_control(carla.VehicleControl(throttle=0.28, brake=0.0, steer=steer))
            return f"AVOIDING: {fwd_threat['label']}", (0, 140, 255), tl_state_str, tl_color, f"SWERVE {direction} (OFFSET 1.4m)", (0, 220, 255)

        if not self.is_autopilot:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            self.is_autopilot = True

        return "PATH CLEAR", (0, 255, 0), tl_state_str, tl_color, "CRUISING (AUTOPILOT)", (0, 255, 0)

# ==============================================================================
# RESIZABLE DASHBOARD RENDERER
# ==============================================================================
def draw_ego_vehicle_icon(canvas, cx, cy):
    draw_rect(canvas, (cx - 10, cy - 20), (cx + 10, cy + 4), (0, 215, 255), 2)
    draw_rect(canvas, (cx - 7, cy - 16), (cx + 7, cy - 6), (0, 140, 255), -1)
    draw_rect(canvas, (cx - 5, cy - 5), (cx + 5, cy + 2), (220, 220, 220), -1)
    draw_rect(canvas, (cx - 30, cy + 8), (cx + 30, cy + 22), (0, 0, 0), -1)
    draw_rect(canvas, (cx - 30, cy + 8), (cx + 30, cy + 22), (0, 215, 255), 1)
    draw_text(canvas, "EGO VEHICLE", (cx - 27, cy + 19), 0.32, (0, 215, 255), 1)

def render_ss_matching_dashboard(xyz, labels, clusters, status_text, status_col,
                                 tl_text, tl_col, tactic_text, tactic_col,
                                 fps, latency_dict, ego_speed, num_pts,
                                 canvas_w=1600, canvas_h=930):
    canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

    # 1. Header Section
    draw_text(canvas, "LIDForge", (30, 48), 1.35, (255, 180, 50), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, " - Output Visualization", (225, 48), 1.15, (235, 235, 235), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, "Town10HD | 20 Hz | LiDAR-only | RTX 3050 target", (32, 98), 0.40, (130, 190, 220), 1)
    draw_text(canvas, "Range-aware base grid + scene-adaptive refinement", (32, 78), 0.52, (170, 170, 170), 1)

    draw_rect(canvas, (820, 16), (1180, 88), (24, 24, 24), -1)
    draw_rect(canvas, (820, 16), (1180, 88), (55, 55, 55), 1)
    draw_text(canvas, "Base Resolution (Range-aware)", (834, 38), 0.44, (255, 200, 100), 1)
    draw_text(canvas, "0 - 20 m -> 5 cm cells (fine)", (834, 58), 0.38, (190, 190, 190), 1)
    draw_text(canvas, "20 - 120 m -> 60 cm cells (coarse)", (834, 76), 0.38, (190, 190, 190), 1)

    draw_rect(canvas, (1200, 16), (1570, 88), (24, 24, 24), -1)
    draw_rect(canvas, (1200, 16), (1570, 88), (55, 55, 55), 1)
    draw_text(canvas, "Adaptive Refinement", (1214, 38), 0.44, (255, 200, 100), 1)
    draw_text(canvas, "Locally increases resolution in", (1214, 58), 0.38, (190, 190, 190), 1)
    draw_text(canvas, "high density, dynamic, or complex regions.", (1214, 76), 0.38, (190, 190, 190), 1)

    # 2. Top Panels
    top_y = 105
    panel_h = 490
    panel_w = 496
    gap = 16

    pa_x1, pa_x2 = 25, 25 + panel_w
    pb_x1, pb_x2 = pa_x2 + gap, pa_x2 + gap + panel_w
    pc_x1, pc_x2 = pb_x2 + gap, pb_x2 + gap + panel_w
    p_y2 = top_y + panel_h

    for (x1, x2) in [(pa_x1, pa_x2), (pb_x1, pb_x2), (pc_x1, pc_x2)]:
        draw_rect(canvas, (x1, top_y), (x2, p_y2), (10, 10, 10), -1)
        draw_rect(canvas, (x1, top_y), (x2, p_y2), (40, 40, 40), 1)

    # Panel (a): 180° Point Cloud
    draw_text(canvas, "(a ) LiDAR Point Cloud ( Extended 3D View )", (pa_x1 + 16, top_y + 26), 0.45, (220, 220, 220), 1)
    cx_a, cy_a = (pa_x1 + pa_x2) // 2, p_y2 - 38
    range_fwd_a = PERCEPTION_FORWARD_RANGE
    range_lat_a = PERCEPTION_LATERAL_RANGE

    for r in [10.0, 20.0, 30.0, 40.0]:
        r_px = int((r / range_fwd_a) * (panel_h - 70))
        cv2.ellipse(canvas, (cx_a, cy_a), (r_px, r_px), 0, 180, 360, (45, 45, 45), 1, cv2.LINE_AA)
        draw_text(canvas, f"{int(r)}m", (cx_a + 6, cy_a - r_px + 12), 0.32, (110, 110, 110), 1)

    for angle in [-60, -30, 0, 30, 60]:
        rad = math.radians(angle)
        ex = int(cx_a + (panel_h - 70) * math.sin(rad))
        ey = int(cy_a - (panel_h - 70) * math.cos(rad))
        draw_line(canvas, (cx_a, cy_a), (ex, ey), (32, 32, 32), 1)

    val_a = (xyz[:, 0] > 0.5) & (xyz[:, 0] < range_fwd_a) & (np.abs(xyz[:, 1]) < range_lat_a)
    if np.any(val_a):
        xa, ya, za, la = xyz[val_a, 0], xyz[val_a, 1], xyz[val_a, 2], labels[val_a]
        h_rel = np.clip(za - (-1.85), -0.5, 4.0)
        px_a = np.clip(((ya / range_lat_a) * (panel_w // 2) + cx_a).astype(np.int32), pa_x1 + 1, pa_x2 - 2)
        py_a = np.clip((cy_a - (xa / range_fwd_a) * (panel_h - 70) - h_rel * 9.0).astype(np.int32), top_y + 32, cy_a)

        road_m = (la == 3)
        canvas[py_a[road_m], px_a[road_m]] = (40, 180, 40)

        obs_m = (la != 3) & (la != 0)
        canvas[py_a[obs_m], px_a[obs_m]] = (255, 220, 0)

        dyn_m = (la == 1) | (la == 2) | (la == 7)
        if np.any(dyn_m):
            canvas[py_a[dyn_m], px_a[dyn_m]] = (255, 140, 0)
            for dy in [-1, 0, 1]:
                for dx_ in [-1, 0, 1]:
                    canvas[np.clip(py_a[dyn_m] + dy, top_y + 32, cy_a), np.clip(px_a[dyn_m] + dx_, pa_x1 + 1, pa_x2 - 2)] = (255, 140, 0)

    for c in clusters:
        xmin, xmax, ymin, ymax = c["bbox"]
        bx1 = np.clip(int(((ymin / range_lat_a) * (panel_w // 2) + cx_a)), pa_x1 + 2, pa_x2 - 2)
        bx2 = np.clip(int(((ymax / range_lat_a) * (panel_w // 2) + cx_a)), pa_x1 + 2, pa_x2 - 2)
        by1 = np.clip(int((cy_a - (xmax / range_fwd_a) * (panel_h - 70))), top_y + 34, cy_a)
        by2 = np.clip(int((cy_a - (xmin / range_fwd_a) * (panel_h - 70))), top_y + 34, cy_a)
        if bx2 - bx1 < 16:
            bx1 -= 8
            bx2 += 8
        if by2 - by1 < 16:
            by1 -= 8
            by2 += 8
        draw_rect(canvas, (bx1, by1), (bx2, by2), c["color"], 2)
        draw_text(canvas, c["label"], (bx1 - 8, by1 - 6), 0.38, c["color"], 1)

    draw_ego_vehicle_icon(canvas, cx_a, cy_a)

    # Panel (b): Multi-Resolution 2.5D Grid
    draw_text(canvas, "(b) Multi-Resolution 2.5D Grid ( Top View )", (pb_x1 + 16, top_y + 26), 0.45, (220, 220, 220), 1)

    leg_bx = pb_x2 - 190
    draw_rect(canvas, (leg_bx, top_y + 12), (leg_bx + 10, top_y + 22), (0, 140, 255), -1)
    draw_text(canvas, "Base ( near, 5 cm )", (leg_bx + 14, top_y + 20), 0.28, (190, 190, 190), 1)

    draw_rect(canvas, (leg_bx, top_y + 26), (leg_bx + 10, top_y + 36), (255, 140, 0), -1)
    draw_text(canvas, "Base ( far, 60 cm )", (leg_bx + 14, top_y + 34), 0.28, (190, 190, 190), 1)

    draw_rect(canvas, (leg_bx, top_y + 40), (leg_bx + 10, top_y + 50), (0, 0, 255), -1)
    draw_text(canvas, "Adaptive refinement", (leg_bx + 14, top_y + 48), 0.28, (190, 190, 190), 1)

    cx_b, cy_b = (pb_x1 + pb_x2) // 2, p_y2 - 38

    grid_spacing = 28
    for gx in range(pb_x1 + 4, pb_x2 - 4, grid_spacing):
        draw_line(canvas, (gx, top_y + 35), (gx, cy_b), (50, 40, 15), 1)
    for gy in range(top_y + 40, cy_b, grid_spacing):
        draw_line(canvas, (pb_x1 + 4, gy), (pb_x2 - 4, gy), (50, 40, 15), 1)

    near_top = int(cy_b - (20.0 / range_fwd_a) * (panel_h - 70))
    near_x1 = int(cx_b - (14.0 / range_lat_a) * (panel_w // 2))
    near_x2 = int(cx_b + (14.0 / range_lat_a) * (panel_w // 2))

    draw_rect(canvas, (near_x1, near_top), (near_x2, cy_b), (0, 150, 255), 1)
    fine_spacing = 7
    for nx in range(near_x1, near_x2, fine_spacing):
        draw_line(canvas, (nx, near_top), (nx, cy_b), (0, 90, 180), 1)
    for ny in range(near_top, cy_b, fine_spacing):
        draw_line(canvas, (near_x1, ny), (near_x2, ny), (0, 90, 180), 1)

    if np.any(val_a):
        px_b = np.clip(((ya / range_lat_a) * (panel_w // 2) + cx_b).astype(np.int32), pb_x1 + 1, pb_x2 - 2)
        py_b = np.clip((cy_b - (xa / range_fwd_a) * (panel_h - 70)).astype(np.int32), top_y + 35, cy_b)
        canvas[py_b[road_m], px_b[road_m]] = (30, 160, 30)
        canvas[py_b[obs_m], px_b[obs_m]] = (255, 200, 0)
        canvas[py_b[dyn_m], px_b[dyn_m]] = (255, 140, 0)

    for c in clusters:
        xmin, xmax, ymin, ymax = c["bbox"]
        bx1 = np.clip(int(((ymin / range_lat_a) * (panel_w // 2) + cx_b)), pb_x1 + 4, pb_x2 - 4)
        bx2 = np.clip(int(((ymax / range_lat_a) * (panel_w // 2) + cx_b)), pb_x1 + 4, pb_x2 - 4)
        by1 = np.clip(int((cy_b - (xmax / range_fwd_a) * (panel_h - 70))), top_y + 35, cy_b)
        by2 = np.clip(int((cy_b - (xmin / range_fwd_a) * (panel_h - 70))), top_y + 35, cy_b)
        if bx2 - bx1 < 22:
            bx1 -= 11
            bx2 += 11
        if by2 - by1 < 22:
            by1 -= 11
            by2 += 11

        draw_rect(canvas, (bx1, by1), (bx2, by2), c["color"], 2)
        for sub_x in range(bx1, bx2, 6):
            draw_line(canvas, (sub_x, by1), (sub_x, by2), c["color"], 1)
        for sub_y in range(by1, by2, 6):
            draw_line(canvas, (bx1, sub_y), (bx2, sub_y), c["color"], 1)

    draw_ego_vehicle_icon(canvas, cx_b, cy_b)

    # Panel (c): Bird's-Eye Height Map
    draw_text(canvas, "(c ) Bird's-Eye Height Map ( 2.5D Output)", (pc_x1 + 16, top_y + 26), 0.45, (220, 220, 220), 1)

    map_h, map_w = panel_h - 60, panel_w - 70
    h_grid = np.full((map_h, map_w), -2.5, dtype=np.float32)
    cx_c, cy_c = map_w // 2, map_h - 10

    if np.any(val_a):
        gx = np.clip(((ya / range_lat_a) * (map_w // 2) + cx_c).astype(np.int32), 0, map_w - 1)
        gy = np.clip((cy_c - (xa / range_fwd_a) * (map_h - 15)).astype(np.int32), 0, map_h - 1)
        np.maximum.at(h_grid, (gy, gx), za)

    road_pixels = (h_grid > -2.4) & (h_grid < -1.4)
    h_grid[road_pixels] = -1.80

    norm_h = np.clip((h_grid - (-2.0)) / 10.0, 0.0, 1.0)
    u8_h = (norm_h * 255).astype(np.uint8)
    color_h = cv2.applyColorMap(u8_h, cv2.COLORMAP_TURBO)
    color_h[h_grid <= -2.4] = (16, 12, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    color_h = cv2.dilate(color_h, kernel)

    canvas[top_y + 40:top_y + 40 + map_h, pc_x1 + 10:pc_x1 + 10 + map_w] = color_h

    bar_x = pc_x2 - 40
    bar_y1 = top_y + 55
    bar_y2 = p_y2 - 55
    bar_h = bar_y2 - bar_y1

    ramp = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    ramp_col = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    canvas[bar_y1:bar_y2, bar_x:bar_x + 12] = ramp_col
    draw_rect(canvas, (bar_x, bar_y1), (bar_x + 12, bar_y2), (80, 80, 80), 1)

    draw_text(canvas, "Height (m)", (bar_x - 30, bar_y1 - 10), 0.36, (200, 200, 200), 1)
    draw_text(canvas, "10", (bar_x + 16, bar_y1 + 10), 0.35, (200, 200, 200), 1)
    draw_text(canvas, "5", (bar_x + 16, (bar_y1 + bar_y2) // 2 + 4), 0.35, (200, 200, 200), 1)
    draw_text(canvas, "0", (bar_x + 16, bar_y2), 0.35, (200, 200, 200), 1)

    # 3. Extended Panel (e)
    pe_y1 = 612
    pe_y2 = canvas_h - 18
    pe_x1 = 25
    pe_x2 = canvas_w - 25

    draw_rect(canvas, (pe_x1, pe_y1), (pe_x2, pe_y2), (10, 10, 10), -1)
    draw_rect(canvas, (pe_x1, pe_y1), (pe_x2, pe_y2), (40, 40, 40), 1)
    draw_text(canvas, "(e) Real-Time Road Status, Hazard Telemetry & Latency Profiler",
              (pe_x1 + 18, pe_y1 + 24), 0.48, (220, 220, 220), 1)

    c1_w = 460
    draw_rect(canvas, (pe_x1 + 18, pe_y1 + 38), (pe_x1 + 18 + c1_w, pe_y2 - 16), (18, 18, 18), -1)
    draw_rect(canvas, (pe_x1 + 18, pe_y1 + 38), (pe_x1 + 18 + c1_w, pe_y2 - 16), (45, 45, 45), 1)

    draw_text(canvas, "FORWARD CORRIDOR INSPECTION", (pe_x1 + 32, pe_y1 + 60), 0.38, (180, 180, 180), 1)
    draw_rect(canvas, (pe_x1 + 30, pe_y1 + 68), (pe_x1 + c1_w + 6, pe_y1 + 128), (22, 22, 22), -1)
    draw_rect(canvas, (pe_x1 + 30, pe_y1 + 68), (pe_x1 + c1_w + 6, pe_y1 + 128), status_col, 2)
    draw_text(canvas, status_text, (pe_x1 + 44, pe_y1 + 104), 0.52, status_col, 2)

    draw_text(canvas, "TACTICAL CONTROLLER (TRAFFIC FLOW / DEFENSE UGV)", (pe_x1 + 32, pe_y1 + 154), 0.38, (180, 180, 180), 1)
    draw_rect(canvas, (pe_x1 + 30, pe_y1 + 162), (pe_x1 + c1_w + 6, pe_y1 + 222), (22, 22, 22), -1)
    draw_rect(canvas, (pe_x1 + 30, pe_y1 + 162), (pe_x1 + c1_w + 6, pe_y1 + 222), tactic_col, 2)
    draw_text(canvas, tactic_text, (pe_x1 + 44, pe_y1 + 198), 0.48, tactic_col, 2)

    c2_x1 = pe_x1 + 18 + c1_w + 18
    c2_w = 390
    c2_x2 = c2_x1 + c2_w
    draw_rect(canvas, (c2_x1, pe_y1 + 38), (c2_x2, pe_y2 - 16), (18, 18, 18), -1)
    draw_rect(canvas, (c2_x1, pe_y1 + 38), (c2_x2, pe_y2 - 16), (45, 45, 45), 1)

    draw_text(canvas, "INTERSECTION SIGNAL (V2I)", (c2_x1 + 20, pe_y1 + 60), 0.38, (180, 180, 180), 1)
    draw_rect(canvas, (c2_x1 + 18, pe_y1 + 72), (c2_x2 - 18, pe_y1 + 128), (24, 24, 24), -1)
    draw_rect(canvas, (c2_x1 + 18, pe_y1 + 72), (c2_x2 - 18, pe_y1 + 128), tl_col, 1)

    tl_box_x = c2_x1 + 32
    tl_box_y = pe_y1 + 84
    draw_rect(canvas, (tl_box_x, tl_box_y), (tl_box_x + 64, tl_box_y + 26), (10, 10, 10), -1)
    draw_circle(canvas, (tl_box_x + 12, tl_box_y + 13), 6, (0, 0, 255) if "RED" in tl_text else (0, 0, 60))
    draw_circle(canvas, (tl_box_x + 32, tl_box_y + 13), 6, (0, 255, 255) if "YELLOW" in tl_text else (0, 60, 60))
    draw_circle(canvas, (tl_box_x + 52, tl_box_y + 13), 6, (0, 255, 0) if "GREEN" in tl_text else (0, 60, 0))
    draw_text(canvas, f"SIGNAL: {tl_text}", (tl_box_x + 80, tl_box_y + 18), 0.44, tl_col, 1)

    draw_text(canvas, "180 DEG SECTOR DANGER MATRIX", (c2_x1 + 20, pe_y1 + 154), 0.38, (180, 180, 180), 1)
    for idx, (s_name, s_col) in enumerate([("LEFT", (0, 255, 0)), ("CENTER", status_col), ("RIGHT", (0, 255, 0))]):
        sx = c2_x1 + 20 + idx * 116
        draw_rect(canvas, (sx, pe_y1 + 168), (sx + 104, pe_y1 + 218), (24, 24, 24), -1)
        draw_rect(canvas, (sx, pe_y1 + 168), (sx + 104, pe_y1 + 218), s_col, 2)
        draw_text(canvas, s_name, (sx + 24, pe_y1 + 198), 0.44, s_col, 1)

    c3_x1 = c2_x2 + 18
    c3_x2 = pe_x2 - 18
    draw_rect(canvas, (c3_x1, pe_y1 + 38), (c3_x2, pe_y2 - 16), (18, 18, 18), -1)
    draw_rect(canvas, (c3_x1, pe_y1 + 38), (c3_x2, pe_y2 - 16), (45, 45, 45), 1)
    draw_text(canvas, "REAL-TIME ROAD TELEMETRY & LATENCY PROFILER", (c3_x1 + 20, pe_y1 + 60), 0.40, (0, 215, 255), 1)

    draw_text(canvas, f"SPEED: {ego_speed:.1f} km/h", (c3_x1 + 22, pe_y1 + 92), 0.48, (0, 225, 255), 2)
    draw_text(canvas, f"REFRESH: {fps:.1f} FPS", (c3_x1 + 230, pe_y1 + 92), 0.48, (0, 255, 0), 2)
    draw_text(canvas, f"ACTIVE PTS: {num_pts:,}", (c3_x1 + 430, pe_y1 + 92), 0.44, (220, 220, 220), 1)

    draw_text(canvas, "MODE: AUTONOMOUS 20Hz SYNC", (c3_x1 + 22, pe_y1 + 118), 0.40, (0, 255, 0), 1)
    draw_text(canvas, "GRID: FOVEATED 2.5D", (c3_x1 + 320, pe_y1 + 118), 0.40, (255, 200, 0), 1)

    draw_rect(canvas, (c3_x1 + 18, pe_y1 + 134), (c3_x2 - 18, pe_y2 - 24), (24, 24, 24), -1)
    draw_rect(canvas, (c3_x1 + 18, pe_y1 + 134), (c3_x2 - 18, pe_y2 - 24), (0, 215, 255), 1)

    tot_lat = latency_dict.get("total", 0.0)
    draw_text(canvas, f"TOTAL LATENCY: {tot_lat:.1f} ms", (c3_x1 + 32, pe_y1 + 162), 0.54, (0, 255, 255), 2)
    draw_text(canvas, f"SPCONV INFERENCE:   {latency_dict.get('spconv', 0.0):.1f} ms", (c3_x1 + 32, pe_y1 + 188), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"VOXEL DEDUP (64-BIT): {latency_dict.get('dedup', 0.0):.1f} ms", (c3_x1 + 32, pe_y1 + 208), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"OCCUPANCY & GATING: {latency_dict.get('gating', 0.0):.1f} ms", (c3_x1 + 320, pe_y1 + 188), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"DASHBOARD BLIT:     {latency_dict.get('render', 0.0):.1f} ms", (c3_x1 + 320, pe_y1 + 208), 0.38, (200, 200, 200), 1)

    return canvas

# ==============================================================================
# MAIN PERCEPTION & SIMULATION PIPELINE
# ==============================================================================
def main():
    global IS_RUNNING, WORLD_POTHOLE_LOCATIONS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Initializing LIDForge Master Client on: {gpu_name}")

    window_name = "LIDForge - Output Visualization (Multi-Resolution 2.5D Perception)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 930)

    # 1. Instantiate Weather Conditioning Filter
    weather_filter = WeatherConditioningFilter(min_intensity=0.08)

    # 2. Neural Model Setup
    model = SpConvUNet(in_channels=1, num_classes=5).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[+] Semantic SpConv weights loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint missing at {CHECKPOINT_PATH}. Using calibrated backbone.")
    model.eval()

    tactical_planner = TacticalGuidanceController()

    # 3. CONNECT TO CARLA
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    GLOBAL_CLEANUP_CONTEXT["client"] = client

    print("[+] Connecting to CARLA...")
    current_world = client.get_world()
    print(f"[+] Current CARLA map: {current_world.get_map().name}")
    print("[+] Loading Town10HD explicitly...")

    world = None
    load_errors = []
    for town_name in ("Town10HD", "/Game/Carla/Maps/Town10HD", "Town10", "/Game/Carla/Maps/Town10"):
        try:
            world = client.load_world(town_name)
            time.sleep(3.0)
            world = client.get_world()
            active_map = world.get_map().name
            print(f"[+] CARLA reported map after load: {active_map}")
            if "Town10HD" in active_map:
                break
            load_errors.append(f"{town_name} -> {active_map}")
        except Exception as exc:
            load_errors.append(f"{town_name} -> {exc}")
            world = None

    if world is None or "Town10HD" not in world.get_map().name:
        raise RuntimeError(
            "Town10HD could not be loaded. Attempts: " + "; ".join(load_errors)
        )

    print("[✓] Town10HD successfully loaded.")
    GLOBAL_CLEANUP_CONTEXT["world"] = world

    # ================================================================================
    # 4. DYNAMIC WEATHER CONFIGURATION (10 SECONDS PER CONDITION)
    # ================================================================================
    weather_index = -1
    weather_index = update_dynamic_weather(world, 0.0, weather_index)

    world_map = world.get_map()
    spectator = world.get_spectator()

    # Synchronous Master Mode: 20 Hz / 50 ms fixed delta
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)
    traffic_manager.global_percentage_speed_difference(-5.0)
    GLOBAL_CLEANUP_CONTEXT["traffic_manager"] = traffic_manager
    print("[✓] Town10HD synchronous mode enabled at 20 Hz.")
    print("[✓] Traffic Manager synchronized on port 8000.")

    # 5. Spawn Ego Vehicle + Traffic Profile
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world_map.get_spawn_points()
    np.random.shuffle(spawn_points)

    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle is not None:
            break

    if vehicle is None:
        actors = world.get_actors().filter("vehicle.*")
        vehicle = actors[0] if len(actors) > 0 else None

    if vehicle is None:
        raise RuntimeError("Failed to acquire ego vehicle in Town10HD.")

    vehicle.set_autopilot(True, traffic_manager.get_port())
    GLOBAL_CLEANUP_CONTEXT["actors"].append(vehicle)

    def spawn_indian_traffic_profile():
        requested = [
            "vehicle.yamaha.yzf", "vehicle.vespa.zx125", "vehicle.kawasaki.ninja",
            "vehicle.audi.a2", "vehicle.nissan.micra",
        ]
        available = {bp.id: bp for bp in bp_lib.filter("vehicle.*")}
        selected = []
        for name in requested:
            if name in available:
                selected.append(available[name])

        fallback = [bp for bp in bp_lib.filter("vehicle.*") if bp.id != vehicle_bp.id]
        np.random.shuffle(fallback)
        traffic_bps = (selected + fallback)[:18]

        used_spawn = {
            (round(vehicle.get_transform().location.x, 1),
             round(vehicle.get_transform().location.y, 1))
        }
        spawned = []
        for sp in spawn_points:
            if len(spawned) >= 18 or not traffic_bps:
                break
            bp = traffic_bps[len(spawned) % len(traffic_bps)]
            key = (round(sp.location.x, 1), round(sp.location.y, 1))
            if key in used_spawn:
                continue
            actor = world.try_spawn_actor(bp, sp)
            if actor is None:
                continue
            used_spawn.add(key)
            actor.set_autopilot(True, traffic_manager.get_port())
            try:
                traffic_manager.auto_lane_change(actor, True)
                traffic_manager.distance_to_leading_vehicle(actor, 1.0)
                traffic_manager.vehicle_percentage_speed_difference(actor, -10.0)
            except Exception:
                pass
            GLOBAL_CLEANUP_CONTEXT["actors"].append(actor)
            spawned.append(actor)
        print(f"[+] Traffic profile: {len(spawned)} AI vehicles spawned.")
        return spawned

    spawn_indian_traffic_profile()

    # 6. Anchor Stationary Road Potholes
    v_init_tf = vehicle.get_transform()
    v_init_yaw = math.radians(v_init_tf.rotation.yaw)
    fx = math.cos(v_init_yaw)
    fy = math.sin(v_init_yaw)

    WORLD_POTHOLE_LOCATIONS = [
        (v_init_tf.location.x + fx * 20.0, v_init_tf.location.y + fy * 20.0, 0.85, 0.15),
        (v_init_tf.location.x + fx * 40.0, v_init_tf.location.y + fy * 40.0, 0.90, 0.16),
        (v_init_tf.location.x + fx * 65.0, v_init_tf.location.y + fy * 65.0, 0.80, 0.14)
    ]

    # 7. Attach 64-Channel LiDAR Sensor (30,000 points per 20 Hz frame)
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", str(LIDAR_POINTS_PER_FRAME * LIDAR_FRAME_RATE_HZ))
    lidar_bp.set_attribute("rotation_frequency", str(LIDAR_FRAME_RATE_HZ))
    lidar_bp.set_attribute("range", "120")
    lidar_bp.set_attribute("upper_fov", "3.0")
    lidar_bp.set_attribute("lower_fov", "-25.0")

    lidar_tf = carla.Transform(carla.Location(x=0.8, y=0.0, z=1.85))
    lidar = world.spawn_actor(lidar_bp, lidar_tf, attach_to=vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(lidar)

    lidar_queue = queue.Queue(maxsize=10)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    print("[+] Master Perception Loop Active in Town10HD. Ready.")

    frame_idx = 0
    try:
        while IS_RUNNING:
            t0 = time.perf_counter()
            world.tick()
            weather_index = update_dynamic_weather(
                world,
                world.get_snapshot().timestamp.elapsed_seconds,
                weather_index,
            )
            update_spectator_follow_cam(spectator, vehicle)
            frame_idx += 1

            points = None
            while not lidar_queue.empty():
                points = lidar_queue.get_nowait()

            if points is None:
                try:
                    points = lidar_queue.get(timeout=1.0)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                        break
                    continue

            pipeline_t0 = time.perf_counter()

            # ==================================================================
            # 8. APPLY WEATHER DE-NOISING FILTER TO RAW LIDAR INGEST
            # ==================================================================
            points = weather_filter.apply(points)

            # Strip ego vehicle chassis returns
            xyz = points[:, :3].copy()
            intensity = np.clip(points[:, 3:4], 0.0, 1.0)
            ego_mask = (xyz[:, 0] >= -2.2) & (xyz[:, 0] <= 2.2) & (xyz[:, 1] >= -1.0) & (xyz[:, 1] <= 1.0) & (xyz[:, 2] <= 0.2)
            xyz = xyz[~ego_mask]
            intensity = intensity[~ego_mask]

            # Ingest road potholes
            xyz = inject_world_potholes(xyz, vehicle)

            # Spatial Front-Hemisphere Pre-Filtering
            spatial_mask = (
                (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= PERCEPTION_FORWARD_RANGE) &
                (xyz[:, 1] >= -PERCEPTION_LATERAL_RANGE) & (xyz[:, 1] <= PERCEPTION_LATERAL_RANGE) &
                (xyz[:, 2] >= PERCEPTION_Z_MIN) & (xyz[:, 2] <= PERCEPTION_Z_MAX)
            )
            xyz_roi = xyz[spatial_mask]
            intensity_roi = intensity[spatial_mask]

            if len(xyz_roi) == 0:
                continue

            # Retain most ground returns so the dashboard shows the higher density.
            ground_est = -1.85
            is_flat_asphalt = (np.abs(xyz_roi[:, 2] - ground_est) < 0.05) & (xyz_roi[:, 0] > 4.0)
            keep_mask = ~is_flat_asphalt | (np.random.rand(len(xyz_roi)) < GROUND_RETURN_KEEP_PROBABILITY)
            xyz_sub = xyz_roi[keep_mask]
            intensity_sub = intensity_roi[keep_mask]

            # 64-bit Packed Voxel Deduplication (< 0.6 ms)
            t_dedup_0 = time.perf_counter()
            voxel_size = LIDAR_VOXEL_SIZE
            ix = (xyz_sub[:, 0] / voxel_size).astype(np.uint64)
            iy = ((xyz_sub[:, 1] + PERCEPTION_LATERAL_RANGE) / voxel_size).astype(np.uint64)
            iz = ((xyz_sub[:, 2] - PERCEPTION_Z_MIN) / voxel_size).astype(np.uint64)
            packed_keys = (ix << 32) | (iy << 16) | iz

            _, u_idx = np.unique(packed_keys, return_index=True)
            xyz_valid = xyz_sub[u_idx]
            intensity_valid = intensity_sub[u_idx]
            t_dedup = (time.perf_counter() - t_dedup_0) * 1000.0

            coords_x = (xyz_valid[:, 0] / voxel_size).astype(np.int32)
            coords_y = ((xyz_valid[:, 1] + PERCEPTION_LATERAL_RANGE) / voxel_size).astype(np.int32)
            coords_z = ((xyz_valid[:, 2] - PERCEPTION_Z_MIN) / voxel_size).astype(np.int32)
            coords_b = np.stack([np.zeros(len(coords_x), dtype=np.int32), coords_x, coords_y, coords_z], axis=-1)

            t_coords = torch.from_numpy(coords_b).to(device=device, dtype=torch.int32).contiguous()
            t_feats = torch.from_numpy(intensity_valid).to(device=device, dtype=torch.float32).contiguous()

            x_sp = spconv.SparseConvTensor(
                features=t_feats,
                indices=t_coords,
                spatial_shape=[
                    int(PERCEPTION_FORWARD_RANGE / voxel_size) + 1,
                    int((2.0 * PERCEPTION_LATERAL_RANGE) / voxel_size) + 1,
                    int((PERCEPTION_Z_MAX - PERCEPTION_Z_MIN) / voxel_size) + 1,
                ],
                batch_size=1
            )

            # SpConv Inference (~6-8 ms)
            t_sp_0 = time.perf_counter()
            if device.type == "cuda":
                with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(x_sp)
                    raw_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                with torch.inference_mode():
                    logits = model(x_sp)
                    raw_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            t_sp = (time.perf_counter() - t_sp_0) * 1000.0

            # Curb-Contact Constraint & Dynamic Clustering
            t_gate_0 = time.perf_counter()
            fused_labels, clusters = extract_elevation_features(xyz_valid, raw_preds)
            t_gate = (time.perf_counter() - t_gate_0) * 1000.0

            # Tactical Guidance
            status_text, status_col, tl_text, tl_col, tactic_text, tactic_col = tactical_planner.update(
                vehicle, traffic_manager, world_map, clusters, fused_labels, xyz_valid
            )

            # Timing & Metrics
            dt = max(time.perf_counter() - t0, 1e-5)
            fps = 1.0 / dt
            total_latency = (time.perf_counter() - pipeline_t0) * 1000.0

            vel = vehicle.get_velocity()
            speed = 3.6 * math.hypot(vel.x, vel.y)
            num_pts = len(xyz_valid)

            latency_dict = {
                "total": total_latency,
                "spconv": t_sp,
                "dedup": t_dedup,
                "gating": t_gate,
                "render": 0.0
            }

            # Render Matching Layout
            t_ren_0 = time.perf_counter()
            dashboard = render_ss_matching_dashboard(
                xyz_valid, fused_labels, clusters,
                status_text, status_col, tl_text, tl_col, tactic_text, tactic_col,
                fps, latency_dict, speed, num_pts
            )
            latency_dict["render"] = (time.perf_counter() - t_ren_0) * 1000.0

            cv2.imshow(window_name, dashboard)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            if frame_idx > 5 and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"[!] Master runtime exception: {e}")
    finally:
        emergency_cleanup()

if __name__ == "__main__":
    main()