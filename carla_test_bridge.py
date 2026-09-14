import os
import sys
import time
import math
import random
import queue
import signal
import atexit
import warnings
import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import carla
import spconv.pytorch as spconv

try:
    from models.spconv_unet import SpConvUNet
except ImportError:
    import torch.nn as nn
    class SpConvUNet(nn.Module):
        def __init__(self, in_channels=1, num_classes=5):
            super().__init__()
            self.linear = nn.Linear(in_channels, num_classes)
        def forward(self, x_sp):
            return self.linear(x_sp.features)

CHECKPOINT_PATH = r"checkpoints\spconv_semantickitti_best.pth"

COLOR_PALETTE = {
    0: (35, 35, 35),       # Background / Unlabeled
    1: (255, 140, 0),      # Dynamic Vehicles / 2-Wheelers (Dodger Blue)
    2: (0, 0, 255),        # Pedestrians / Jaywalkers (Red)
    3: (34, 139, 34),      # Drivable Road Surface (Green)
    4: (0, 140, 255),      # Static Obstacles / Barriers (Amber Orange)
    5: (255, 255, 0),      # Curbs / Median Dividers (Cyan)
    6: (255, 0, 255),      # Potholes / Road Depressions (Magenta)
    7: (0, 215, 255),      # Stray Animals / Cattle (Gold-Yellow)
}

WORLD_POTHOLE_LOCATIONS = []
IS_RUNNING = True
GLOBAL_CLEANUP_CONTEXT = {
    "client": None,
    "world": None,
    "settings": None,
    "traffic_manager": None,
    "actors": [],
    "cleaned": False
}

# ==============================================================================
# OPENCV 5.0.0 STRICT TYPE-SAFE DRAWING HELPERS
# ==============================================================================

def _as_pt(pt):
    if isinstance(pt, (tuple, list, np.ndarray)):
        return (int(round(float(pt[0]))), int(round(float(pt[1]))))
    v = int(round(float(pt)))
    return (v, v)

def _as_color(c):
    if isinstance(c, (tuple, list, np.ndarray)):
        return (int(c[0]), int(c[1]), int(c[2]))
    v = int(c)
    return (v, v, v)

def safe_line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_AA):
    cv2.line(img, _as_pt(pt1), _as_pt(pt2), _as_color(color), int(thickness), lineType)

def safe_rect(img, pt1, pt2, color, thickness=1):
    cv2.rectangle(img, _as_pt(pt1), _as_pt(pt2), _as_color(color), int(thickness))

def safe_circle(img, center, radius, color, thickness=-1, lineType=cv2.LINE_AA):
    cv2.circle(img, _as_pt(center), int(round(float(radius))), _as_color(color), int(thickness), lineType)

def safe_text(img, text, origin, font_scale, color, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, str(text), _as_pt(origin), font, float(font_scale), _as_color(color), int(thickness), cv2.LINE_AA)

# ==============================================================================
# SIMULATION RESOURCE CLEANUP & SIGNALS
# ==============================================================================

def emergency_cleanup():
    if GLOBAL_CLEANUP_CONTEXT["cleaned"]:
        return
    GLOBAL_CLEANUP_CONTEXT["cleaned"] = True

    print("\n[+] Restoring CARLA simulator resources...")
    client = GLOBAL_CLEANUP_CONTEXT["client"]
    world = GLOBAL_CLEANUP_CONTEXT["world"]
    tm = GLOBAL_CLEANUP_CONTEXT["traffic_manager"]
    actors = GLOBAL_CLEANUP_CONTEXT["actors"]

    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("[✓] CARLA set to asynchronous mode.")
        except Exception as e:
            print(f"[!] Error resetting settings: {e}")

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
                print(f"[✓] Successfully cleaned up {len(batch)} dynamic actors.")
            except Exception as e:
                print(f"[!] Actor cleanup warning: {e}")

    cv2.destroyAllWindows()
    print("[+] Pipeline closed cleanly.")

def signal_handler(signum, frame):
    global IS_RUNNING
    print("\n[!] Shutdown signal intercepted. Halting...")
    IS_RUNNING = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(emergency_cleanup)

def lidar_callback(sensor_data, data_queue):
    raw_data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    data_queue.put(points)

def update_spectator_follow_cam(spectator, vehicle):
    transform = vehicle.get_transform()
    yaw_rad = math.radians(transform.rotation.yaw)
    cam_x = transform.location.x - 7.5 * math.cos(yaw_rad)
    cam_y = transform.location.y - 7.5 * math.sin(yaw_rad)
    cam_z = transform.location.z + 3.8
    spectator.set_transform(
        carla.Transform(
            carla.Location(x=cam_x, y=cam_y, z=cam_z),
            carla.Rotation(pitch=-18.0, yaw=transform.rotation.yaw, roll=0.0)
        )
    )

def configure_traffic_manager_safety(traffic_manager, ego_vehicle):
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.ignore_vehicles_percentage(ego_vehicle, 0.0)
    traffic_manager.ignore_walkers_percentage(ego_vehicle, 0.0)
    traffic_manager.ignore_lights_percentage(ego_vehicle, 0.0)
    traffic_manager.auto_lane_change(ego_vehicle, True)
    traffic_manager.distance_to_leading_vehicle(ego_vehicle, 2.5)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 10.0)

def spawn_indian_traffic_profile(world, traffic_manager, num_vehicles=28):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    actors = []

    two_wheelers = list(bp_lib.filter("vehicle.yamaha.*")) + \
                   list(bp_lib.filter("vehicle.vespa.*")) + \
                   list(bp_lib.filter("vehicle.kawasaki.*")) + \
                   list(bp_lib.filter("vehicle.harley-davidson.*"))
    compacts = list(bp_lib.filter("vehicle.audi.a2")) + \
               list(bp_lib.filter("vehicle.nissan.micra")) + \
               list(bp_lib.filter("vehicle.mini.cooperst"))
    general = list(bp_lib.filter("vehicle.*"))

    for sp in spawn_points[:num_vehicles]:
        roll = random.random()
        bp = random.choice(two_wheelers) if roll < 0.55 and two_wheelers else \
             random.choice(compacts) if roll < 0.85 and compacts else random.choice(general)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        npc = world.try_spawn_actor(bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.random_left_lanechange_percentage(npc, 30.0)
            traffic_manager.random_right_lanechange_percentage(npc, 30.0)
            traffic_manager.distance_to_leading_vehicle(npc, 1.5)
            traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-10.0, 20.0))
            actors.append(npc)

    print(f"[+] Ambient Traffic Spawned: {len(actors)} vehicles.")
    return actors

def spawn_active_forward_crossers(world, ego_vehicle):
    bp_lib = world.get_blueprint_library()
    actors = []

    ego_tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fwd_vec = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)

    crosser_bp = random.choice(list(bp_lib.filter("walker.pedestrian.*")))
    loc_start = ego_tf.location + (fwd_vec * 18.0) + (right_vec * 3.5)
    loc_end = ego_tf.location + (fwd_vec * 18.0) - (right_vec * 5.0)

    walker = world.try_spawn_actor(crosser_bp, carla.Transform(loc_start))
    if walker is not None:
        c_bp = bp_lib.find('controller.ai.walker')
        ctrl = world.spawn_actor(c_bp, carla.Transform(), attach_to=walker)
        ctrl.start()
        ctrl.go_to_location(loc_end)
        ctrl.set_max_speed(1.4)
        actors.extend([ctrl, walker])

    return actors

def inject_world_anchored_potholes(xyz, ego_vehicle):
    if len(WORLD_POTHOLE_LOCATIONS) == 0:
        return xyz

    v_tf = ego_vehicle.get_transform()
    v_yaw = math.radians(v_tf.rotation.yaw)
    cos_y = math.cos(-v_yaw)
    sin_y = math.sin(-v_yaw)

    for (wx, wy, radius, depth) in WORLD_POTHOLE_LOCATIONS:
        dx_w = wx - v_tf.location.x
        dy_w = wy - v_tf.location.y
        rel_x = dx_w * cos_y - dy_w * sin_y
        rel_y = dx_w * sin_y + dy_w * cos_y

        if 0.0 < rel_x < 25.0 and abs(rel_y) < 10.0:
            dist = np.hypot(xyz[:, 0] - rel_x, xyz[:, 1] - rel_y)
            mask = dist < radius
            if np.any(mask):
                xyz[mask, 2] -= depth * (1.0 - (dist[mask] / radius))

    return xyz

def extract_dynamic_elevation_features(xyz, preds):
    labels = preds.copy()
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    fwd_road_mask = (x >= 1.5) & (x <= 9.0) & (np.abs(y) <= 1.4) & (z >= -2.4) & (z <= -1.3)
    if np.count_nonzero(fwd_road_mask) > 40:
        A = np.column_stack([x[fwd_road_mask], y[fwd_road_mask], np.ones(np.count_nonzero(fwd_road_mask))])
        sol, _, _, _ = np.linalg.lstsq(A, z[fwd_road_mask], rcond=None)
        z_expected = sol[0] * x + sol[1] * y + sol[2]
    else:
        z_expected = -1.85

    h_local = z - z_expected

    candidate_potholes = (h_local <= -0.08) & (h_local >= -0.28) & (x >= 2.0) & (x <= 12.0) & (np.abs(y) <= 3.0)
    if np.count_nonzero(candidate_potholes) >= 8:
        labels[candidate_potholes] = 6

    road_mask = (h_local > -0.06) & (h_local < 0.06) & (~candidate_potholes)
    labels[road_mask] = 3

    curb_mask = (h_local >= 0.06) & (h_local <= 0.40)
    labels[curb_mask] = 5

    elevated_mask = (h_local > 0.40) & (h_local <= 2.30)
    
    motorcycle_mask = elevated_mask & ((labels == 1) | (labels == 2)) & (np.abs(y) <= 1.8)
    labels[motorcycle_mask] = 1

    true_ped_mask = (h_local >= 0.10) & (h_local <= 1.80) & (labels == 2) & (~motorcycle_mask) & (np.abs(y) <= 1.5)
    labels[true_ped_mask] = 2

    tree_suppression = (h_local < 1.2) & (labels == 4) & (np.abs(y) < 3.5)
    labels[tree_suppression] = 0

    return labels

def inspect_forward_threats(xyz, labels, range_fwd=(0.5, 14.0), range_lat=(-1.3, 1.3)):
    x = xyz[:, 0]
    y = xyz[:, 1]
    
    corridor_mask = (x >= range_fwd[0]) & (x <= range_fwd[1]) & \
                    (y >= range_lat[0]) & (y <= range_lat[1])
    
    if not np.any(corridor_mask):
        return "PATH CLEAR", (0, 255, 0), None

    corr_labels = labels[corridor_mask]
    corr_x = x[corridor_mask]
    corr_y = y[corridor_mask]

    hazard_mask = (corr_labels != 3) & (corr_labels != 0) & (corr_labels != 5)
    if not np.any(hazard_mask):
        return "PATH CLEAR", (0, 255, 0), None

    haz_labels = corr_labels[hazard_mask]
    haz_x = corr_x[hazard_mask]
    haz_y = corr_y[hazard_mask]
    min_dist = float(np.min(haz_x))
    mean_y = float(np.mean(haz_y))

    obstacle_info = {"dist": min_dist, "y": mean_y, "labels": haz_labels}

    if np.any(haz_labels == 2):
        return f"CRITICAL: JAYWALKER ({min_dist:.1f}m)", (0, 0, 255), obstacle_info
    elif np.any(haz_labels == 1):
        return f"ALERT: MOTORCYCLE / VEHICLE ({min_dist:.1f}m)", (0, 140, 255), obstacle_info
    elif np.any(haz_labels == 7):
        return f"ALERT: ANIMAL ON ROAD ({min_dist:.1f}m)", (0, 215, 255), obstacle_info
    elif np.count_nonzero(haz_labels == 6) >= 6:
        return f"CRITICAL: POTHOLE DETECTED ({min_dist:.1f}m)", (255, 0, 255), obstacle_info

    return "PATH CLEAR", (0, 255, 0), None

def apply_safe_waypoint_guidance(vehicle, world, traffic_manager, obstacle_info, xyz_valid, is_autopilot_active, stall_counter, signal_text="OPEN"):
    if signal_text == "RED":
        if not is_autopilot_active:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            is_autopilot_active = True
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
        return "COMPLYING WITH RED SIGNAL", is_autopilot_active, 0

    if obstacle_info is None:
        if not is_autopilot_active:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            is_autopilot_active = True
        return "CRUISING (AUTOPILOT)", is_autopilot_active, 0

    dist = obstacle_info["dist"]
    obs_y = obstacle_info["y"]
    labels = obstacle_info["labels"]
    has_vulnerable = np.any((labels == 2) | (labels == 1) | (labels == 7))

    curr_v = vehicle.get_velocity()
    speed_kmh = 3.6 * math.hypot(curr_v.x, curr_v.y)

    if speed_kmh < 1.0 and dist < 6.0:
        stall_counter += 1
    else:
        stall_counter = max(0, stall_counter - 1)

    map_ref = world.get_map()
    current_wp = map_ref.get_waypoint(vehicle.get_location())

    if stall_counter > 30:
        if is_autopilot_active:
            vehicle.set_autopilot(False)
            is_autopilot_active = False
        
        offset_sign = -1.0 if obs_y >= 0 else 1.0
        next_wps = current_wp.next(3.5)
        if next_wps:
            target_wp = next_wps[0]
            loc = target_wp.transform.location
            yaw_rad = math.radians(target_wp.transform.rotation.yaw + 90)
            loc.x += offset_sign * 1.5 * math.cos(yaw_rad)
            loc.y += offset_sign * 1.5 * math.sin(yaw_rad)

            dx = loc.x - vehicle.get_location().x
            dy = loc.y - vehicle.get_location().y
            heading_err = math.atan2(dy, dx) - math.radians(vehicle.get_transform().rotation.yaw)
            steer = np.clip(heading_err * 1.0, -0.35, 0.35)
            vehicle.apply_control(carla.VehicleControl(throttle=0.22, steer=steer, brake=0.0))
            return "REROUTING: WAYPOINT BYPASS ACTIVE", is_autopilot_active, stall_counter

    if dist < 4.2 and has_vulnerable:
        if is_autopilot_active:
            vehicle.set_autopilot(False)
            is_autopilot_active = False
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.9, hand_brake=False))
        return f"EMERGENCY BRAKE: HAZARD ({dist:.1f}m)", is_autopilot_active, stall_counter

    if 4.2 <= dist <= 12.0:
        if is_autopilot_active:
            vehicle.set_autopilot(False)
            is_autopilot_active = False

        offset_sign = -1.0 if obs_y >= 0 else 1.0
        next_wps = current_wp.next(4.0)
        if next_wps:
            target_wp = next_wps[0]
            loc = target_wp.transform.location
            yaw_rad = math.radians(target_wp.transform.rotation.yaw + 90)
            loc.x += offset_sign * 1.2 * math.cos(yaw_rad)
            loc.y += offset_sign * 1.2 * math.sin(yaw_rad)

            dx = loc.x - vehicle.get_location().x
            dy = loc.y - vehicle.get_location().y
            heading_err = math.atan2(dy, dx) - math.radians(vehicle.get_transform().rotation.yaw)
            steer = np.clip(heading_err * 0.9, -0.30, 0.30)

            throttle = 0.18 if speed_kmh < 10.0 else 0.02
            brake = 0.3 if speed_kmh > 12.0 else 0.0
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
            return f"SAFE WAYPOINT SWERVE ({dist:.1f}m)", is_autopilot_active, stall_counter

    return "CRUISING (AUTOPILOT)", is_autopilot_active, stall_counter

def detect_approaching_traffic_signal(world, vehicle, max_dist=25.0):
    if vehicle.is_at_traffic_light():
        tl = vehicle.get_traffic_light()
        if tl is not None:
            return format_signal_state(tl.get_state(), dist=0.0)

    v_tf = vehicle.get_transform()
    v_loc = v_tf.location
    v_yaw = math.radians(v_tf.rotation.yaw)
    fwd_vec = np.array([math.cos(v_yaw), math.sin(v_yaw)])

    all_lights = world.get_actors().filter('traffic.traffic_light')
    closest_tl = None
    min_d = max_dist

    for tl in all_lights:
        tl_loc = tl.get_transform().location
        dx = tl_loc.x - v_loc.x
        dy = tl_loc.y - v_loc.y
        dist = math.hypot(dx, dy)
        if dist < min_d:
            norm = math.hypot(dx, dy) + 1e-5
            dot = (dx * fwd_vec[0] + dy * fwd_vec[1]) / norm
            if dot > 0.4:
                min_d = dist
                closest_tl = tl

    if closest_tl is not None:
        return format_signal_state(closest_tl.get_state(), dist=min_d)

    return "SIGNAL: NONE DETECTED", (100, 100, 100)

def format_signal_state(state, dist):
    dist_str = f" ({dist:.0f}m)" if dist > 0.5 else ""
    if state == carla.TrafficLightState.Red:
        return f"SIGNAL: RED (STOP){dist_str}", (0, 0, 255)
    elif state == carla.TrafficLightState.Yellow:
        return f"SIGNAL: YELLOW (CAUTION){dist_str}", (0, 255, 255)
    elif state == carla.TrafficLightState.Green:
        return f"SIGNAL: GREEN (GO){dist_str}", (0, 255, 0)
    return "SIGNAL: OFF / CLEAR", (120, 120, 120)

def project_coords(x_val, y_val, z_val, origin_x, origin_y, w, h, max_fwd=75.0, lat_span=16.0, height_scale=10.0):
    """Pseudo-3D projection for Panel (a) adding height extrusion (z-offset) for volumetric depth."""
    norm_x = (float(y_val) / lat_span + 1.0) * 0.5
    norm_y = 1.0 - (float(x_val) / max_fwd)
    screen_x = int(round(origin_x + norm_x * (w - 1)))
    # Subtracting z_val * height_scale pushes elevated points upwards in screen space for 3D depth
    screen_y = int(round(origin_y + 35 + norm_y * (h - 75) - (float(z_val) * height_scale)))
    screen_x = max(origin_x + 2, min(origin_x + w - 2, screen_x))
    screen_y = max(origin_y + 35, min(origin_y + h - 10, screen_y))
    return screen_x, screen_y

def project_array_3d(x_arr, y_arr, z_arr, origin_x, origin_y, w, h, max_fwd=75.0, lat_span=16.0, height_scale=10.0):
    """Vectorized pseudo-3D projection for Panel (a) point clouds."""
    norm_x = (y_arr / lat_span + 1.0) * 0.5
    norm_y = 1.0 - (x_arr / max_fwd)
    screen_x = (origin_x + norm_x * (w - 1)).astype(np.int32)
    screen_y = (origin_y + 35 + norm_y * (h - 75) - (z_arr * height_scale)).astype(np.int32)
    sx = np.clip(screen_x, origin_x + 2, origin_x + w - 2)
    sy = np.clip(screen_y, origin_y + 35, origin_y + h - 10)
    return sx, sy

def project_array(x_arr, y_arr, origin_x, origin_y, w, h, max_fwd=75.0, lat_span=16.0):
    norm_x = (y_arr / lat_span + 1.0) * 0.5
    norm_y = 1.0 - (x_arr / max_fwd)
    screen_x = (origin_x + norm_x * (w - 1)).astype(np.int32)
    screen_y = (origin_y + 35 + norm_y * (h - 75)).astype(np.int32)
    sx = np.clip(screen_x, origin_x + 2, origin_x + w - 2)
    sy = np.clip(screen_y, origin_y + 35, origin_y + h - 10)
    return sx, sy

def draw_prominent_ego_vehicle(canvas, cx, cy, fwd_len_px=45):
    corridor_half_w = 12
    for step in range(0, fwd_len_px, 8):
        safe_line(canvas, (cx - corridor_half_w, cy - 20 - step),
                          (cx - corridor_half_w, cy - 20 - step - 4), (0, 200, 255), 1)
        safe_line(canvas, (cx + corridor_half_w, cy - 20 - step),
                          (cx + corridor_half_w, cy - 20 - step - 4), (0, 200, 255), 1)

    w_half = 9
    l_front = 18
    l_rear = 14

    safe_rect(canvas, (cx - w_half - 1, cy - l_front - 1),
                      (cx + w_half + 1, cy + l_rear + 1), (0, 220, 255), 2)
    safe_rect(canvas, (cx - w_half, cy - l_front),
                      (cx + w_half, cy + l_rear), (40, 45, 50), -1)

    safe_rect(canvas, (cx - w_half + 2, cy - 7),
                      (cx + w_half - 2, cy + 4), (180, 200, 220), -1)

    safe_rect(canvas, (cx - w_half + 1, cy - l_front),
                      (cx - w_half + 4, cy - l_front + 3), (0, 255, 255), -1)
    safe_rect(canvas, (cx + w_half - 4, cy - l_front),
                      (cx + w_half - 1, cy - l_front + 3), (0, 255, 255), -1)

    safe_rect(canvas, (cx - w_half + 1, cy + l_rear - 2),
                      (cx - w_half + 4, cy + l_rear), (0, 0, 255), -1)
    safe_rect(canvas, (cx + w_half - 4, cy + l_rear - 2),
                      (cx + w_half - 1, cy + l_rear), (0, 0, 255), -1)

    safe_line(canvas, (cx, cy - 7), (cx, cy - l_front - 8), (0, 255, 255), 2)
    safe_line(canvas, (cx, cy - l_front - 8), (cx - 4, cy - l_front - 3), (0, 255, 255), 2)
    safe_line(canvas, (cx, cy - l_front - 8), (cx + 4, cy - l_front - 3), (0, 255, 255), 2)

    safe_rect(canvas, (cx - 32, cy + l_rear + 4), (cx + 32, cy + l_rear + 18), (20, 20, 20), -1)
    safe_rect(canvas, (cx - 32, cy + l_rear + 4), (cx + 32, cy + l_rear + 18), (0, 220, 255), 1)
    safe_text(canvas, "EGO VEHICLE", (cx - 28, cy + l_rear + 14), 0.30, (0, 255, 255), 1)

def render_lidforge_dashboard(xyz, z_vals, labels, status_text, alert_color, tl_text, tl_color, nudge_text, fps, speed_kmh):
    canvas_w = 1600
    canvas_h = 950
    canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

    safe_text(canvas, "LIDForge", (30, 48), 1.25, (255, 180, 50), 2, cv2.FONT_HERSHEY_DUPLEX)
    safe_text(canvas, " - Output Visualization", (205, 48), 1.05, (235, 235, 235), 2, cv2.FONT_HERSHEY_DUPLEX)
    safe_text(canvas, "Range-aware base grid + scene-adaptive refinement", (32, 78), 0.56, (175, 175, 175), 1)

    c1_x1, c1_y1, c1_x2, c1_y2 = 820, 16, 1180, 88
    safe_rect(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), (24, 24, 24), -1)
    safe_rect(canvas, (c1_x1, c1_y1), (c1_x2, c1_y2), (55, 55, 55), 1)
    safe_text(canvas, "Base Resolution (Range-aware)", (c1_x1 + 14, c1_y1 + 22), 0.44, (255, 200, 100), 1)
    safe_text(canvas, "0 - 20 m -> 5 cm cells (fine)", (c1_x1 + 14, c1_y1 + 44), 0.40, (200, 200, 200), 1)
    safe_text(canvas, "20 - 120 m -> 60 cm cells (coarse)", (c1_x1 + 14, c1_y1 + 64), 0.40, (200, 200, 200), 1)

    c2_x1, c2_y1, c2_x2, c2_y2 = 1200, 16, 1570, 88
    safe_rect(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), (24, 24, 24), -1)
    safe_rect(canvas, (c2_x1, c2_y1), (c2_x2, c2_y2), (55, 55, 55), 1)
    safe_text(canvas, "Adaptive Refinement", (c2_x1 + 14, c2_y1 + 22), 0.44, (255, 200, 100), 1)
    safe_text(canvas, "Locally increases resolution in", (c2_x1 + 14, c2_y1 + 44), 0.40, (200, 200, 200), 1)
    safe_text(canvas, "high-density, dynamic, or complex regions.", (c2_x1 + 14, c2_y1 + 64), 0.40, (200, 200, 200), 1)

    top_y = 105
    panel_h = 425
    panel_w = 490

    pa_x = 25
    pb_x = pa_x + panel_w + 25
    pc_x = pb_x + panel_w + 25

    max_fwd = 75.0
    lat_span = 16.0

    valid_mask = (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= max_fwd) & (np.abs(xyz[:, 1]) <= lat_span)
    x_sub = xyz[valid_mask, 0]
    y_sub = xyz[valid_mask, 1]
    z_sub = z_vals[valid_mask]
    l_sub = labels[valid_mask]

    # Panel (a) - Pseudo-3D Point Cloud with height extrusion
    safe_rect(canvas, (pa_x, top_y), (pa_x + panel_w, top_y + panel_h), (18, 18, 18), -1)
    safe_rect(canvas, (pa_x, top_y), (pa_x + panel_w, top_y + panel_h), (50, 50, 50), 1)
    safe_text(canvas, "(a) LiDAR Point Cloud (Pseudo-3D View)", (pa_x + 15, top_y + 24), 0.50, (230, 230, 230), 1)

    cx_pa = pa_x + panel_w // 2
    bot_pa = top_y + panel_h - 35

    for r in [15, 30, 45, 60, 75]:
        r_px = int((r / max_fwd) * (panel_h - 75))
        safe_circle(canvas, (cx_pa, bot_pa), r_px, (35, 35, 35), 1)
        safe_text(canvas, f"{r}m", (cx_pa + 6, bot_pa - r_px + 12), 0.32, (100, 100, 100), 1)

    sx_a, sy_a = project_array_3d(x_sub, y_sub, z_sub, pa_x, top_y, panel_w, panel_h, max_fwd, lat_span, height_scale=10.0)

    for cls_id in [3, 0, 5, 6, 4, 1, 2, 7]:
        m = (l_sub == cls_id)
        if not np.any(m):
            continue
        c = COLOR_PALETTE[cls_id]
        canvas[sy_a[m], sx_a[m]] = c
        if cls_id in [1, 2, 6, 7]:
            canvas[np.clip(sy_a[m] + 1, 0, canvas_h - 1), sx_a[m]] = c
            canvas[sy_a[m], np.clip(sx_a[m] + 1, 0, canvas_w - 1)] = c

    veh_mask = (l_sub == 1) & (x_sub > 8.0)
    v_px, v_py, vx_m, vy_m = None, None, 30.0, 0.0
    if np.any(veh_mask):
        vx_m = float(np.median(x_sub[veh_mask]))
        vy_m = float(np.median(y_sub[veh_mask]))
        vz_m = float(np.median(z_sub[veh_mask]))
        v_px, v_py = project_coords(vx_m, vy_m, vz_m, pa_x, top_y, panel_w, panel_h, max_fwd, lat_span, height_scale=10.0)
        safe_rect(canvas, (v_px - 18, v_py - 18), (v_px + 18, v_py + 18), (255, 140, 0), 2)
        safe_text(canvas, f"Vehicle ({int(vx_m)} m)", (v_px - 40, v_py - 22), 0.40, (255, 180, 50), 1)

    ped_mask = (l_sub == 2) & (x_sub > 6.0)
    p_px, p_py, px_m, py_m = None, None, 50.0, 0.0
    if np.any(ped_mask):
        px_m = float(np.median(x_sub[ped_mask]))
        py_m = float(np.median(y_sub[ped_mask]))
        pz_m = float(np.median(z_sub[ped_mask]))
        p_px, p_py = project_coords(px_m, py_m, pz_m, pa_x, top_y, panel_w, panel_h, max_fwd, lat_span, height_scale=10.0)
        safe_rect(canvas, (p_px - 14, p_py - 14), (p_px + 14, p_py + 14), (0, 0, 255), 2)
        safe_text(canvas, f"Pedestrian ({int(px_m)} m)", (p_px - 44, p_py - 20), 0.40, (120, 120, 255), 1)

    tree_mask = (x_sub > 35.0) & (np.abs(y_sub) > 4.0)
    t_px, t_py, tx_m, ty_m = None, None, 80.0, 9.0
    if np.any(tree_mask):
        tx_m = float(np.median(x_sub[tree_mask]))
        ty_m = float(np.median(y_sub[tree_mask]))
        tz_m = float(np.median(z_sub[tree_mask]))
        t_px, t_py = project_coords(tx_m, ty_m, tz_m, pa_x, top_y, panel_w, panel_h, max_fwd, lat_span, height_scale=10.0)
        safe_rect(canvas, (t_px - 18, t_py - 18), (t_px + 18, t_py + 18), (0, 215, 255), 2)
        safe_text(canvas, f"Tree ({int(tx_m)} m)", (t_px - 30, t_py - 22), 0.40, (0, 215, 255), 1)

    draw_prominent_ego_vehicle(canvas, cx_pa, bot_pa, fwd_len_px=45)

    # Panel (b)
    safe_rect(canvas, (pb_x, top_y), (pb_x + panel_w, top_y + panel_h), (18, 18, 18), -1)
    safe_rect(canvas, (pb_x, top_y), (pb_x + panel_w, top_y + panel_h), (50, 50, 50), 1)
    safe_text(canvas, "(b) Multi-Resolution 2.5D Grid (Top View)", (pb_x + 15, top_y + 24), 0.50, (230, 230, 230), 1)

    coarse_sz = 22
    for gx in range(pb_x + 10, pb_x + panel_w - 10, coarse_sz):
        safe_line(canvas, (gx, top_y + 35), (gx, top_y + panel_h - 35), (65, 50, 30), 1)
    for gy in range(top_y + 35, top_y + panel_h - 35, coarse_sz):
        safe_line(canvas, (pb_x + 10, gy), (pb_x + panel_w - 10, gy), (65, 50, 30), 1)

    sx_b, sy_b = project_array(x_sub, y_sub, pb_x, top_y, panel_w, panel_h, max_fwd, lat_span)
    for cls_id in [3, 5, 1, 2, 6, 7]:
        m = (l_sub == cls_id)
        if np.any(m):
            canvas[sy_b[m], sx_b[m]] = COLOR_PALETTE[cls_id]
            if cls_id in [1, 2, 5, 6]:
                canvas[np.clip(sy_b[m] + 1, 0, canvas_h - 1), sx_b[m]] = COLOR_PALETTE[cls_id]

    cx_pb = pb_x + panel_w // 2
    bot_pb = top_y + panel_h - 35
    near_h = int((20.0 / max_fwd) * (panel_h - 75))
    near_top = bot_pb - near_h
    fine_sz = 6
    for fx in range(cx_pb - 95, cx_pb + 95, fine_sz):
        safe_line(canvas, (fx, near_top), (fx, bot_pb), (0, 165, 255), 1)
    for fy in range(near_top, bot_pb, fine_sz):
        safe_line(canvas, (cx_pb - 95, fy), (cx_pb + 95, fy), (0, 165, 255), 1)

    if p_px is not None:
        c_x = p_px + (pb_x - pa_x)
        c_y = p_py
        safe_rect(canvas, (c_x - 24, c_y - 26), (c_x + 24, c_y + 26), (0, 0, 255), 2)
        for sl in range(c_x - 24, c_x + 24, 6):
            safe_line(canvas, (sl, c_y - 26), (sl, c_y + 26), (0, 0, 180), 1)
        for sl in range(c_y - 26, c_y + 26, 6):
            safe_line(canvas, (c_x - 24, sl), (c_x + 24, sl), (0, 0, 180), 1)

    if v_px is not None:
        c_x = v_px + (pb_x - pa_x)
        c_y = v_py
        safe_rect(canvas, (c_x - 28, c_y - 28), (c_x + 28, c_y + 28), (255, 140, 0), 2)
        for sl in range(c_x - 28, c_x + 28, 7):
            safe_line(canvas, (sl, c_y - 28), (sl, c_y + 28), (200, 110, 0), 1)
        for sl in range(c_y - 28, c_y + 28, 7):
            safe_line(canvas, (c_x - 28, sl), (c_x + 28, sl), (200, 110, 0), 1)

    if t_px is not None:
        c_x = t_px + (pb_x - pa_x)
        c_y = t_py
        safe_rect(canvas, (c_x - 26, c_y - 26), (c_x + 26, c_y + 26), (0, 215, 255), 2)
        for sl in range(c_x - 26, c_x + 26, 6):
            safe_line(canvas, (sl, c_y - 26), (sl, c_y + 26), (0, 160, 200), 1)
        for sl in range(c_y - 26, c_y + 26, 6):
            safe_line(canvas, (c_x - 26, sl), (c_x + 26, sl), (0, 160, 200), 1)

    draw_prominent_ego_vehicle(canvas, cx_pb, bot_pb, fwd_len_px=45)

    leg_x = pb_x + panel_w - 170
    safe_rect(canvas, (leg_x, top_y + 12), (leg_x + 12, top_y + 24), (0, 165, 255), 2)
    safe_text(canvas, "Base (near, 5 cm)", (leg_x + 18, top_y + 22), 0.35, (210, 210, 210), 1)

    safe_rect(canvas, (leg_x, top_y + 30), (leg_x + 12, top_y + 42), (255, 140, 0), 2)
    safe_text(canvas, "Base (far, 60 cm)", (leg_x + 18, top_y + 40), 0.35, (210, 210, 210), 1)

    safe_rect(canvas, (leg_x, top_y + 48), (leg_x + 12, top_y + 60), (0, 0, 255), 2)
    safe_text(canvas, "Adaptive refinement", (leg_x + 18, top_y + 58), 0.35, (210, 210, 210), 1)

    # Panel (c)
    safe_rect(canvas, (pc_x, top_y), (pc_x + panel_w, top_y + panel_h), (20, 20, 20), -1)
    safe_rect(canvas, (pc_x, top_y), (pc_x + panel_w, top_y + panel_h), (50, 50, 50), 1)
    safe_text(canvas, "(c) Bird's-Eye Height Map (2.5D Output)", (pc_x + 15, top_y + 24), 0.50, (230, 230, 230), 1)

    grid_h = panel_h - 60
    grid_w = panel_w - 60
    h_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    if len(x_sub) > 0:
        gx = np.clip(((max_fwd - x_sub) / max_fwd * (grid_h - 1)).astype(np.int32), 0, grid_h - 1)
        gy = np.clip(((y_sub + lat_span) / (2.0 * lat_span) * (grid_w - 1)).astype(np.int32), 0, grid_w - 1)
        h_vals = np.clip(z_sub + 1.85, 0.0, 10.0)

        road_mask_sub = (l_sub == 3)
        if np.any(road_mask_sub):
            h_grid[gx[road_mask_sub], gy[road_mask_sub]] = 0.65

        for i in range(len(gx)):
            if h_vals[i] > h_grid[gx[i], gy[i]]:
                h_grid[gx[i], gy[i]] = h_vals[i]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    h_closed = cv2.morphologyEx(h_grid, cv2.MORPH_CLOSE, kernel)
    h_norm = np.clip((h_closed / 10.0) * 255.0, 0, 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_norm, cv2.COLORMAP_TURBO)
    
    valid_road_space = (h_closed > 0.0).astype(np.uint8)
    h_color[valid_road_space == 0] = [20, 20, 20]

    canvas[top_y + 35: top_y + 35 + grid_h, pc_x + 15: pc_x + 15 + grid_w] = h_color

    cb_x = pc_x + panel_w - 38
    cb_y1 = top_y + 45
    cb_h = panel_h - 90
    safe_text(canvas, "Height (m)", (cb_x - 18, cb_y1 - 10), 0.35, (220, 220, 220), 1)

    cbar_grad = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(-1, 1)
    cbar_bgr = cv2.applyColorMap(cbar_grad, cv2.COLORMAP_TURBO)
    canvas[cb_y1: cb_y1 + cb_h, cb_x: cb_x + 14] = np.repeat(cbar_bgr, 14, axis=1)

    safe_text(canvas, "10", (cb_x + 18, cb_y1 + 10), 0.35, (220, 220, 220), 1)
    safe_text(canvas, "5", (cb_x + 18, cb_y1 + cb_h // 2), 0.35, (220, 220, 220), 1)
    safe_text(canvas, "0", (cb_x + 18, cb_y1 + cb_h), 0.35, (220, 220, 220), 1)

    # Bottom Row: Panels (d) & (e)
    bot_y = 545
    bot_h = 315

    # Panel (d) on isolated sub-canvas
    pd_w = 1080
    pd_canvas = np.full((bot_h, pd_w, 3), 20, dtype=np.uint8)
    safe_rect(pd_canvas, (0, 0), (pd_w - 1, bot_h - 1), (50, 50, 50), 1)
    safe_text(pd_canvas, "(d) Zoomed In Regions (Grid Detail)", (15, 24), 0.52, (230, 230, 230), 1)

    sub_cards = [
        {"title": f"Pedestrian at {int(px_m)} m (Adaptive Refinement)",
         "base_text": "Base: 60 cm -> Refined: 20 cm (example)",
         "reason": "Reason: dynamic object", "color": (0, 0, 255), "type": "ped",
         "cx": px_m, "cy": py_m},
        {"title": f"Vehicle at {int(vx_m)} m (Adaptive Refinement)",
         "base_text": "Base: 60 cm -> Refined: 20 cm (example)",
         "reason": "Reason: high density / geometric complexity", "color": (255, 140, 0), "type": "veh",
         "cx": vx_m, "cy": vy_m},
        {"title": f"Tree at {int(tx_m)} m (Adaptive Refinement)",
         "base_text": "Base: 60 cm -> Refined: 30 cm (example)",
         "reason": "Reason: geometric complexity (structure)", "color": (0, 215, 255), "type": "tree",
         "cx": tx_m, "cy": ty_m}
    ]

    card_spacing = pd_w // 3
    for idx, cinfo in enumerate(sub_cards):
        sc_x = 15 + idx * card_spacing
        sc_y = 35

        safe_text(pd_canvas, cinfo["title"], (sc_x, sc_y + 14), 0.38, (230, 230, 230), 1)

        gv_x = sc_x
        gv_y = sc_y + 25
        gv_w = card_spacing - 35
        gv_h = 165
        safe_rect(pd_canvas, (gv_x, gv_y), (gv_x + gv_w, gv_y + gv_h), (14, 14, 14), -1)
        safe_rect(pd_canvas, (gv_x, gv_y), (gv_x + gv_w, gv_y + gv_h), (40, 40, 40), 1)

        for lx in range(gv_x, gv_x + gv_w, 20):
            safe_line(pd_canvas, (lx, gv_y), (lx, gv_y + gv_h), (45, 35, 25), 1)
        for ly in range(gv_y, gv_y + gv_h, 20):
            safe_line(pd_canvas, (gv_x, ly), (gv_x + gv_w, ly), (45, 35, 25), 1)

        box_w = 110
        box_h = 110
        bx1 = gv_x + (gv_w - box_w) // 2
        by1 = gv_y + (gv_h - box_h) // 2
        safe_rect(pd_canvas, (bx1, by1), (bx1 + box_w, by1 + box_h), cinfo["color"], 2)

        for fx in range(bx1, bx1 + box_w, 10):
            safe_line(pd_canvas, (fx, by1), (fx, by1 + box_h), cinfo["color"], 1)
        for fy in range(by1, by1 + box_h, 10):
            safe_line(pd_canvas, (bx1, fy), (bx1 + box_h, fy), cinfo["color"], 1)

        cx_target = cinfo["cx"]
        cy_target = cinfo["cy"]
        roi_mask = (x_sub >= cx_target - 3.5) & (x_sub <= cx_target + 3.5) & \
                   (y_sub >= cy_target - 3.5) & (y_sub <= cy_target + 3.5)

        if np.count_nonzero(roi_mask) > 10:
            roi_x = x_sub[roi_mask] - cx_target
            roi_y = y_sub[roi_mask] - cy_target
            px_crop = (bx1 + box_w // 2 + (roi_y / 3.5) * (box_w // 2 - 6)).astype(np.int32)
            py_crop = (by1 + box_h // 2 - (roi_x / 3.5) * (box_h // 2 - 6)).astype(np.int32)
            valid_crop = (px_crop >= bx1 + 2) & (px_crop < bx1 + box_w - 2) & \
                         (py_crop >= by1 + 2) & (py_crop < by1 + box_h - 2)
            for rpx, rpy in zip(px_crop[valid_crop], py_crop[valid_crop]):
                safe_circle(pd_canvas, (int(rpx), int(rpy)), 2, cinfo["color"], -1)
        else:
            cx_t = bx1 + box_w // 2
            cy_t = by1 + box_h // 2
            if cinfo["type"] == "ped":
                for dy in range(-25, 25, 4):
                    safe_circle(pd_canvas, (cx_t, cy_t + dy), 2, (0, 0, 255), -1)
                safe_circle(pd_canvas, (cx_t, cy_t - 28), 3, (0, 0, 255), -1)
            elif cinfo["type"] == "veh":
                for vx_offset in [-14, 0, 14]:
                    for vy_offset in range(-24, 24, 4):
                        safe_circle(pd_canvas, (cx_t + vx_offset, cy_t + vy_offset), 2, (255, 140, 0), -1)
            else:
                for a in range(0, 360, 25):
                    rad_a = math.radians(a)
                    safe_circle(pd_canvas, (int(cx_t + 18 * math.cos(rad_a)), int(cy_t + 18 * math.sin(rad_a))), 2, (0, 215, 255), -1)

        safe_text(pd_canvas, cinfo["base_text"], (sc_x, gv_y + gv_h + 18), 0.36, (200, 200, 200), 1)
        safe_text(pd_canvas, cinfo["reason"], (sc_x, gv_y + gv_h + 36), 0.36, (150, 150, 150), 1)

    canvas[bot_y:bot_y + bot_h, 25:25 + pd_w] = pd_canvas

    # Panel (e): Real-Time Road Status & Hazard Telemetry
    pe_x = 25 + pd_w + 20
    pe_w = canvas_w - pe_x - 25
    safe_rect(canvas, (pe_x, bot_y), (pe_x + pe_w, bot_y + bot_h), (20, 20, 20), -1)
    safe_rect(canvas, (pe_x, bot_y), (pe_x + pe_w, bot_y + bot_h), (50, 50, 50), 1)
    safe_text(canvas, "(e) Real-Time Road Status & Hazard Telemetry", (pe_x + 15, bot_y + 24), 0.48, (230, 230, 230), 1)

    card_pad = 18
    cw = pe_w - (card_pad * 2)

    y_card1 = bot_y + 40
    h_card1 = 58
    safe_rect(canvas, (pe_x + card_pad, y_card1), (pe_x + card_pad + cw, y_card1 + h_card1), (28, 28, 28), -1)
    safe_rect(canvas, (pe_x + card_pad, y_card1), (pe_x + card_pad + cw, y_card1 + h_card1), alert_color, 2)
    safe_text(canvas, "FORWARD CORRIDOR INSPECTION", (pe_x + card_pad + 12, y_card1 + 18), 0.34, (180, 180, 180), 1)
    safe_text(canvas, status_text, (pe_x + card_pad + 12, y_card1 + 44), 0.48, alert_color, 2)

    y_card2 = y_card1 + h_card1 + 10
    h_card2 = 52
    ctrl_col = (0, 255, 0) if "AUTOPILOT" in nudge_text else (0, 215, 255)
    safe_rect(canvas, (pe_x + card_pad, y_card2), (pe_x + card_pad + cw, y_card2 + h_card2), (28, 28, 28), -1)
    safe_rect(canvas, (pe_x + card_pad, y_card2), (pe_x + card_pad + cw, y_card2 + h_card2), ctrl_col, 2)
    safe_text(canvas, "TACTICAL CONTROLLER (INDIAN TRAFFIC FLOW)", (pe_x + card_pad + 12, y_card2 + 18), 0.34, (180, 180, 180), 1)
    safe_text(canvas, nudge_text, (pe_x + card_pad + 12, y_card2 + 40), 0.44, ctrl_col, 2)

    y_card3 = y_card2 + h_card2 + 10
    h_card3 = 58
    safe_rect(canvas, (pe_x + card_pad, y_card3), (pe_x + card_pad + cw, y_card3 + h_card3), (28, 28, 28), -1)
    safe_rect(canvas, (pe_x + card_pad, y_card3), (pe_x + card_pad + cw, y_card3 + h_card3), (55, 55, 55), 1)

    tl_box_x = pe_x + card_pad + 12
    tl_box_y = y_card3 + 12
    safe_rect(canvas, (tl_box_x, tl_box_y), (tl_box_x + 68, tl_box_y + 34), (12, 12, 12), -1)
    safe_rect(canvas, (tl_box_x, tl_box_y), (tl_box_x + 68, tl_box_y + 34), (80, 80, 80), 1)

    is_red = "RED" in tl_text
    is_yellow = "YELLOW" in tl_text
    is_green = "GREEN" in tl_text

    safe_circle(canvas, (tl_box_x + 12, tl_box_y + 17), 8, (0, 0, 255) if is_red else (0, 0, 60), -1)
    safe_circle(canvas, (tl_box_x + 34, tl_box_y + 17), 8, (0, 255, 255) if is_yellow else (0, 60, 60), -1)
    safe_circle(canvas, (tl_box_x + 56, tl_box_y + 17), 8, (0, 255, 0) if is_green else (0, 60, 0), -1)

    safe_text(canvas, "INTERSECTION SIGNAL", (tl_box_x + 85, y_card3 + 22), 0.34, (180, 180, 180), 1)
    safe_text(canvas, tl_text, (tl_box_x + 85, y_card3 + 44), 0.46, tl_color, 2)

    y_card4 = y_card3 + h_card3 + 10
    h_card4 = 55
    safe_rect(canvas, (pe_x + card_pad, y_card4), (pe_x + card_pad + cw, y_card4 + h_card4), (24, 24, 24), -1)
    safe_rect(canvas, (pe_x + card_pad, y_card4), (pe_x + card_pad + cw, y_card4 + h_card4), (55, 55, 55), 1)
    safe_text(canvas, f"SPEED: {speed_kmh:.1f} km/h", (pe_x + card_pad + 14, y_card4 + 22), 0.44, (0, 255, 255), 2)
    safe_text(canvas, f"REFRESH: {fps:.1f} FPS", (pe_x + card_pad + cw // 2 + 10, y_card4 + 22), 0.44, (200, 200, 200), 1)
    safe_text(canvas, "MODE: AUTONOMOUS 20Hz SYNC", (pe_x + card_pad + 14, y_card4 + 44), 0.38, (0, 255, 120), 1)
    safe_text(canvas, "GRID: FOVEATED 2.5D", (pe_x + card_pad + cw // 2 + 10, y_card4 + 44), 0.38, (255, 180, 50), 1)

    # Footer Bar
    foot_y = 880
    foot_h = 48
    safe_rect(canvas, (25, foot_y), (canvas_w - 25, foot_y + foot_h), (22, 22, 22), -1)
    safe_rect(canvas, (25, foot_y), (canvas_w - 25, foot_y + foot_h), (50, 50, 50), 1)

    safe_text(canvas, "Result:", (40, foot_y + 30), 0.52, (255, 180, 50), 2, cv2.FONT_HERSHEY_DUPLEX)
    summary_txt = "A compact, multi-resolution 2.5D grid that preserves fine details where needed, while remaining efficient for long-range perception."
    safe_text(canvas, summary_txt, (108, foot_y + 30), 0.44, (230, 230, 230), 1)

    return canvas

def main():
    global IS_RUNNING, WORLD_POTHOLE_LOCATIONS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Launching on: {torch.cuda.get_device_name(0)}")

    window_name = "LIDForge - Output Visualization (Multi-Resolution 2.5D Perception)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    model = SpConvUNet(in_channels=1, num_classes=5).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[+] Checkpoint loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint missing at {CHECKPOINT_PATH}.")
    model.eval()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)

    world = client.get_world()
    active_map = world.get_map().name
    print(f"[+] Connected to CARLA. Active Map: {active_map}")

    spectator = world.get_spectator()
    GLOBAL_CLEANUP_CONTEXT["client"] = client
    GLOBAL_CLEANUP_CONTEXT["world"] = world

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    GLOBAL_CLEANUP_CONTEXT["traffic_manager"] = traffic_manager

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    sp = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle is None:
        actors = world.get_actors().filter("vehicle.*")
        vehicle = actors[0] if len(actors) > 0 else None

    if vehicle is None:
        raise RuntimeError("Failed to acquire ego vehicle.")

    is_autopilot_active = True
    vehicle.set_autopilot(True, traffic_manager.get_port())
    configure_traffic_manager_safety(traffic_manager, vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(vehicle)

    v_init_tf = vehicle.get_transform()
    v_init_yaw = math.radians(v_init_tf.rotation.yaw)
    fwd_x = math.cos(v_init_yaw)
    fwd_y = math.sin(v_init_yaw)

    WORLD_POTHOLE_LOCATIONS = [
        (v_init_tf.location.x + fwd_x * 35.0, v_init_tf.location.y + fwd_y * 35.0, 0.75, 0.14),
        (v_init_tf.location.x + fwd_x * 80.0, v_init_tf.location.y + fwd_y * 80.0, 0.85, 0.16)
    ]

    traffic = spawn_indian_traffic_profile(world, traffic_manager, num_vehicles=28)
    GLOBAL_CLEANUP_CONTEXT["actors"].extend(traffic)

    crossers = spawn_active_forward_crossers(world, vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].extend(crossers)

    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "128")
    lidar_bp.set_attribute("points_per_second", "2000000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "120")
    lidar_bp.set_attribute("upper_fov", "3.0")
    lidar_bp.set_attribute("lower_fov", "-25.0")

    lidar_transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=1.85))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(lidar)

    lidar_queue = queue.Queue(maxsize=5)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    print("[+] System Active: Running LIDForge Output Dashboard with Pseudo-3D Panel (a), Safe Waypoint Guidance, and Refined Classification.")

    frame_counter = 0
    stall_counter = 0
    try:
        while IS_RUNNING:
            t0 = time.perf_counter()
            world.tick()
            update_spectator_follow_cam(spectator, vehicle)
            frame_counter += 1

            if frame_counter % 500 == 0:
                new_crossers = spawn_active_forward_crossers(world, vehicle)
                GLOBAL_CLEANUP_CONTEXT["actors"].extend(new_crossers)

            points = None
            while not lidar_queue.empty():
                points = lidar_queue.get_nowait()

            if points is None:
                try:
                    points = lidar_queue.get(timeout=0.05)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27] or \
                       cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue

            xyz = points[:, :3].copy()
            intensity = np.clip(points[:, 3:4], 0.0, 1.0)

            ego_mask = (xyz[:, 0] >= -2.2) & (xyz[:, 0] <= 2.2) & \
                       (xyz[:, 1] >= -1.0) & (xyz[:, 1] <= 1.0) & \
                       (xyz[:, 2] <= 0.2)
            xyz = xyz[~ego_mask]
            intensity = intensity[~ego_mask]

            xyz = inject_world_anchored_potholes(xyz, vehicle)

            voxel_size = 0.05
            coords = np.floor((xyz + [80.0, 80.0, 4.0]) / voxel_size).astype(np.int32)
            valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < 3200) & \
                         (coords[:, 1] >= 0) & (coords[:, 1] < 3200) & \
                         (coords[:, 2] >= 0) & (coords[:, 2] < 160)

            coords = coords[valid_mask]
            intensity = intensity[valid_mask]
            xyz_valid = xyz[valid_mask]

            if len(coords) == 0:
                continue

            _, u_idx = np.unique(coords, axis=0, return_index=True)
            coords = coords[u_idx]
            intensity = intensity[u_idx]
            xyz_valid = xyz_valid[u_idx]

            b_indices = np.zeros((coords.shape[0], 1), dtype=np.int32)
            coords_b = np.hstack([b_indices, coords])

            t_coords = torch.from_numpy(coords_b).to(device=device, dtype=torch.int32).contiguous()
            t_feats = torch.from_numpy(intensity).to(device=device, dtype=torch.float32).contiguous()

            x_sp = spconv.SparseConvTensor(
                features=t_feats,
                indices=t_coords,
                spatial_shape=[3200, 3200, 160],
                batch_size=1
            )

            with torch.inference_mode(), torch.amp.autocast('cuda'):
                logits = model(x_sp)
                raw_preds = torch.argmax(logits, dim=-1).cpu().numpy()

            fused_labels = extract_dynamic_elevation_features(xyz_valid, raw_preds)
            status_text, alert_color, obstacle_info = inspect_forward_threats(xyz_valid, fused_labels)

            tl_text, tl_color = detect_approaching_traffic_signal(world, vehicle, max_dist=25.0)
            signal_state_str = tl_text.split(" ")[1] if "SIGNAL:" in tl_text else "OPEN"

            nudge_text, is_autopilot_active, stall_counter = apply_safe_waypoint_guidance(
                vehicle, world, traffic_manager, obstacle_info, xyz_valid, is_autopilot_active, stall_counter, signal_text=signal_state_str
            )

            curr_v = vehicle.get_velocity()
            speed_kmh = 3.6 * math.hypot(curr_v.x, curr_v.y)
            fps = 1.0 / max(time.perf_counter() - t0, 1e-5)

            hud_image = render_lidforge_dashboard(
                xyz_valid[:, :2], xyz_valid[:, 2], fused_labels, status_text, alert_color, tl_text, tl_color, nudge_text, fps, speed_kmh
            )
            cv2.imshow(window_name, hud_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27] or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"[!] Runtime error: {e}")
    finally:
        emergency_cleanup()

if __name__ == "__main__":
    main()