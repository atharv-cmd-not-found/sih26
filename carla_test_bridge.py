import os
import sys
import time
import math
import random
import signal
import atexit
import warnings
import queue
import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import carla
import spconv.pytorch as spconv
from models.spconv_unet import SpConvUNet

CHECKPOINT_PATH = r"checkpoints\spconv_semantickitti_best.pth"

# Semantic Color Palette (BGR)
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
    "traffic_manager": None,
    "actors": [],
    "cleaned": False
}

def emergency_cleanup():
    """Restores CARLA settings and batch destroys dynamic actors."""
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

def configure_traffic_manager_resilience(traffic_manager, ego_vehicle):
    """Configures tight follow distance and aggressive lane negotiation."""
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(0.8)
    traffic_manager.auto_lane_change(ego_vehicle, True)
    traffic_manager.distance_to_leading_vehicle(ego_vehicle, 0.8)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, -10.0)
    traffic_manager.ignore_vehicles_percentage(ego_vehicle, 25.0)

def spawn_indian_traffic_profile(world, traffic_manager, num_vehicles=28):
    """Spawns 50%+ 2-wheelers, auto-rickshaw compacts, and aggressive lane changers."""
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
            traffic_manager.random_left_lanechange_percentage(npc, 40.0)
            traffic_manager.random_right_lanechange_percentage(npc, 40.0)
            traffic_manager.distance_to_leading_vehicle(npc, 0.7)
            traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(-10.0, 25.0))
            actors.append(npc)

    print(f"[+] Ambient Traffic Spawned: {len(actors)} vehicles (High 2-Wheeler Density).")
    return actors

def spawn_active_forward_crossers(world, ego_vehicle):
    """Spawns dynamic pedestrians and strays crossing perpendicularly across the path."""
    bp_lib = world.get_blueprint_library()
    actors = []

    ego_tf = ego_vehicle.get_transform()
    yaw_rad = math.radians(ego_tf.rotation.yaw)
    fwd_vec = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
    right_vec = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)

    # 1. Jaywalker crossing roadway from right to left
    crosser_bp = random.choice(list(bp_lib.filter("walker.pedestrian.*")))
    loc_start = ego_tf.location + (fwd_vec * 20.0) + (right_vec * 4.5)
    loc_end = ego_tf.location + (fwd_vec * 20.0) - (right_vec * 6.0)

    walker = world.try_spawn_actor(crosser_bp, carla.Transform(loc_start))
    if walker is not None:
        c_bp = bp_lib.find('controller.ai.walker')
        ctrl = world.spawn_actor(c_bp, carla.Transform(), attach_to=walker)
        ctrl.start()
        ctrl.go_to_location(loc_end)
        ctrl.set_max_speed(1.4)
        actors.extend([ctrl, walker])

    # 2. Stray Animal Profile crossing further ahead
    stray_bps = list(bp_lib.filter("walker.pedestrian.0010")) or list(bp_lib.filter("walker.pedestrian.*"))
    stray_start = ego_tf.location + (fwd_vec * 32.0) - (right_vec * 4.0)
    stray_end = ego_tf.location + (fwd_vec * 32.0) + (right_vec * 5.5)
    stray = world.try_spawn_actor(random.choice(stray_bps), carla.Transform(stray_start))
    if stray is not None:
        c_bp = bp_lib.find('controller.ai.walker')
        ctrl = world.spawn_actor(c_bp, carla.Transform(), attach_to=stray)
        ctrl.start()
        ctrl.go_to_location(stray_end)
        ctrl.set_max_speed(1.1)
        actors.extend([ctrl, stray])

    return actors

def inject_world_anchored_potholes(xyz, ego_vehicle):
    """Carves asphalt depressions at fixed world coordinates."""
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
    """
    PITCH/ROLL INVARIANT ELEVATION CLASSIFICATION:
    Computes a plane fit across the vehicle's immediate forward lane so braking and
    stopping at red lights does not cause normal asphalt to register as a pothole.
    """
    labels = preds.copy()
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # Robust local road plane fitting in the immediate driving corridor
    fwd_road_mask = (x >= 1.5) & (x <= 9.0) & (np.abs(y) <= 1.4) & (z >= -2.4) & (z <= -1.3)
    if np.count_nonzero(fwd_road_mask) > 40:
        A = np.column_stack([x[fwd_road_mask], y[fwd_road_mask], np.ones(np.count_nonzero(fwd_road_mask))])
        sol, _, _, _ = np.linalg.lstsq(A, z[fwd_road_mask], rcond=None)
        z_expected = sol[0] * x + sol[1] * y + sol[2]
    else:
        z_expected = -1.85

    h_local = z - z_expected

    # 1. Potholes: Strict depression threshold (-8cm to -28cm) restricted to near field
    candidate_potholes = (h_local <= -0.08) & (h_local >= -0.28) & (x >= 2.0) & (x <= 12.0) & (np.abs(y) <= 3.0)
    if np.count_nonzero(candidate_potholes) >= 8:
        labels[candidate_potholes] = 6

    # 2. Drivable Road Bed
    road_mask = (h_local > -0.06) & (h_local < 0.06) & (~candidate_potholes)
    labels[road_mask] = 3

    # 3. Curbs / Median Dividers (+6cm to +40cm)
    curb_mask = (h_local >= 0.06) & (h_local <= 0.40)
    labels[curb_mask] = 5

    # 4. Elevated objects (> 40cm)
    elevated_mask = (h_local > 0.40) & (h_local <= 2.30)
    rider_mask = elevated_mask & (labels == 2) & (h_local >= 0.65)
    labels[rider_mask] = 1

    true_ped_mask = (h_local >= 0.10) & (h_local <= 1.90) & (labels == 2) & (~rider_mask)
    labels[true_ped_mask] = 2

    # Low-slung stray animals
    animal_mask = (h_local >= 0.15) & (h_local <= 0.85) & (labels == 4) & (np.abs(y) <= 3.0)
    labels[animal_mask] = 7

    labels[elevated_mask & (labels == 1)] = 1
    labels[elevated_mask & (labels != 1) & (labels != 2) & (labels != 7)] = 4

    return labels

def inspect_forward_threats(xyz, labels, range_fwd=(0.8, 11.0), range_lat=(-1.35, 1.35)):
    """Inspects immediate forward bumper safety corridor."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    
    corridor_mask = (x >= range_fwd[0]) & (x <= range_fwd[1]) & \
                    (y >= range_lat[0]) & (y <= range_lat[1])
    
    if not np.any(corridor_mask):
        return "PATH CLEAR", (0, 255, 0), None

    corr_labels = labels[corridor_mask]
    corr_x = x[corridor_mask]
    corr_y = y[corridor_mask]

    hazard_mask = (corr_labels != 3) & (corr_labels != 0)
    if not np.any(hazard_mask):
        return "PATH CLEAR", (0, 255, 0), None

    haz_labels = corr_labels[hazard_mask]
    haz_x = corr_x[hazard_mask]
    haz_y = corr_y[hazard_mask]
    min_dist = np.min(haz_x)
    mean_y = np.mean(haz_y)

    obstacle_info = {"dist": min_dist, "y": mean_y, "labels": haz_labels}

    if np.count_nonzero(haz_labels == 6) >= 6:
        return f"CRITICAL: POTHOLE DETECTED ({min_dist:.1f}m)", (255, 0, 255), obstacle_info
    elif np.any(haz_labels == 2):
        return f"CRITICAL: JAYWALKER CROSSING ({min_dist:.1f}m)", (0, 0, 255), obstacle_info
    elif np.any(haz_labels == 7):
        return f"ALERT: STRAY ANIMAL ON ROAD ({min_dist:.1f}m)", (0, 215, 255), obstacle_info
    elif np.any(haz_labels == 1):
        return f"WARNING: VEHICLE / 2-WHEELER ({min_dist:.1f}m)", (0, 140, 255), obstacle_info
    elif np.any(haz_labels == 5):
        return f"ALERT: ROAD MEDIAN / CURB ({min_dist:.1f}m)", (255, 255, 0), obstacle_info
    elif np.any(haz_labels == 4):
        return f"CAUTION: ROAD BARRIER / OBSTACLE ({min_dist:.1f}m)", (0, 165, 255), obstacle_info

    return "PATH CLEAR", (0, 255, 0), None

def apply_reactive_nudge_control(vehicle, traffic_manager, obstacle_info, xyz_valid, is_autopilot_active):
    """
    INDIAN DRIVING REACTIVE CONTROLLER:
    Maneuvers the ego vehicle smoothly around obstructions onto clear asphalt.
    Tracks autopilot state using an explicit boolean flag.
    """
    if obstacle_info is None:
        if not is_autopilot_active:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            is_autopilot_active = True
        return "CRUISING (AUTOPILOT)", is_autopilot_active

    dist = obstacle_info["dist"]
    obs_y = obstacle_info["y"]
    labels = obstacle_info["labels"]

    # Emergency full brake if obstacle is immediately at the front bumper (< 2.2m)
    if dist < 2.2 and np.any((labels == 2) | (labels == 7)):
        if is_autopilot_active:
            vehicle.set_autopilot(False)
            is_autopilot_active = False
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=True))
        return "EMERGENCY BRAKE (CLOSE PROXIMITY)", is_autopilot_active

    # Active Nudge / Overtake Zone (2.2m to 10.0m)
    if 2.2 <= dist <= 10.0:
        if is_autopilot_active:
            vehicle.set_autopilot(False)
            is_autopilot_active = False

        # Lateral clearance check (-Y is Left, +Y is Right in CARLA)
        left_space = (xyz_valid[:, 0] >= 1.0) & (xyz_valid[:, 0] <= 8.0) & (xyz_valid[:, 1] < -1.4)
        right_space = (xyz_valid[:, 0] >= 1.0) & (xyz_valid[:, 0] <= 8.0) & (xyz_valid[:, 1] > 1.4)

        if obs_y >= 0:  # Obstacle is on right -> steer left
            target_steer = -0.32 if np.count_nonzero(left_space) < 100 else -0.22
            action = f"NUDGING LEFT AROUND OBSTACLE ({dist:.1f}m)"
        else:           # Obstacle is on left -> steer right
            target_steer = 0.32 if np.count_nonzero(right_space) < 100 else 0.22
            action = f"NUDGING RIGHT AROUND OBSTACLE ({dist:.1f}m)"

        curr_v = vehicle.get_velocity()
        speed_kmh = 3.6 * math.hypot(curr_v.x, curr_v.y)
        throttle = 0.30 if speed_kmh < 15.0 else 0.05
        brake = 0.25 if speed_kmh > 18.0 else 0.0

        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=target_steer, brake=brake))
        return action, is_autopilot_active

    return "CRUISING (AUTOPILOT)", is_autopilot_active

def detect_approaching_traffic_signal(world, vehicle, max_dist=25.0):
    """Detects traffic light state affecting the vehicle's driving path."""
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

def render_dual_clipmap_hud(points, labels, status_text, alert_color, tl_text, tl_color, nudge_text, fps):
    panel_w, panel_h = 450, 450
    legend_bar_h = 48
    canvas_w = panel_w * 2
    canvas_h = panel_h + legend_bar_h

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    near_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    far_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    center = panel_w // 2

    for p_img, max_r, rings in [(near_panel, 20.0, [5, 10, 15, 20]), (far_panel, 120.0, [30, 60, 90, 120])]:
        for r in rings:
            r_px = int((r / max_r) * center)
            cv2.circle(p_img, (center, center), r_px, (45, 45, 45), 1)
            cv2.putText(p_img, f"{r}m", (center + 4, center - r_px + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (110, 110, 110), 1)
        cv2.line(p_img, (center, 0), (center, panel_h), (30, 30, 30), 1)
        cv2.line(p_img, (0, center), (panel_w, center), (30, 30, 30), 1)

    x = points[:, 0]
    y = points[:, 1]

    # --- 1. Near View (0-20m) ---
    mask_near = (np.abs(x) < 20.0) & (np.abs(y) < 20.0)
    if np.any(mask_near):
        px_n = ((y[mask_near] / 20.0 + 1.0) * 0.5 * (panel_w - 1)).astype(np.int32)
        py_n = (((-x[mask_near]) / 20.0 + 1.0) * 0.5 * (panel_h - 1)).astype(np.int32)
        px_n = np.clip(px_n, 0, panel_w - 1)
        py_n = np.clip(py_n, 0, panel_h - 1)
        lbl_n = labels[mask_near]

        rd = (lbl_n == 3)
        if np.any(rd):
            near_panel[py_n[rd], px_n[rd]] = COLOR_PALETTE[3]

        ph = (lbl_n == 6)
        if np.any(ph):
            for rx, ry in zip(px_n[ph], py_n[ph]):
                cv2.circle(near_panel, (rx, ry), 3, COLOR_PALETTE[6], -1)

        cb = (lbl_n == 5)
        if np.any(cb):
            for rx, ry in zip(px_n[cb], py_n[cb]):
                cv2.circle(near_panel, (rx, ry), 2, COLOR_PALETTE[5], -1)

        for c_id, rad in [(4, 2), (1, 3), (2, 3), (7, 3)]:
            m = (lbl_n == c_id)
            if np.any(m):
                for rx, ry in zip(px_n[m], py_n[m]):
                    cv2.circle(near_panel, (rx, ry), rad, COLOR_PALETTE[c_id], -1)

    # --- 2. Far View (0-120m) ---
    mask_far = (np.abs(x) < 120.0) & (np.abs(y) < 120.0)
    if np.any(mask_far):
        px_f = ((y[mask_far] / 120.0 + 1.0) * 0.5 * (panel_w - 1)).astype(np.int32)
        py_f = (((-x[mask_far]) / 120.0 + 1.0) * 0.5 * (panel_h - 1)).astype(np.int32)
        px_f = np.clip(px_f, 0, panel_w - 1)
        py_f = np.clip(py_f, 0, panel_h - 1)
        lbl_f = labels[mask_far]

        rd_f = (lbl_f == 3)
        if np.any(rd_f):
            far_panel[py_f[rd_f], px_f[rd_f]] = COLOR_PALETTE[3]

        for c_id in [6, 5, 4, 1, 2, 7]:
            m = (lbl_f == c_id)
            if np.any(m):
                far_panel[py_f[m], px_f[m]] = COLOR_PALETTE[c_id]

    cv2.rectangle(near_panel, (center - 5, center - 11), (center + 5, center + 11), (0, 255, 255), -1)
    cv2.circle(far_panel, (center, center), 3, (0, 255, 255), -1)

    # Forward Corridor Safety Box
    fwd_min_py = int(((-11.0 / 20.0 + 1.0) * 0.5 * (panel_h - 1)))
    fwd_max_py = int(((-0.8 / 20.0 + 1.0) * 0.5 * (panel_h - 1)))
    fwd_min_px = int(((-1.35 / 20.0 + 1.0) * 0.5 * (panel_w - 1)))
    fwd_max_px = int(((1.35 / 20.0 + 1.0) * 0.5 * (panel_w - 1)))
    cv2.rectangle(near_panel, (fwd_min_px, fwd_min_py), (fwd_max_px, fwd_max_py), (100, 100, 100), 1)

    canvas[0:panel_h, 0:panel_w] = near_panel
    canvas[0:panel_h, panel_w:canvas_w] = far_panel

    # Panel Headers
    cv2.putText(canvas, "NEAR: 0-20m (Steps & Potholes)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
    cv2.putText(canvas, f"FAR: 0-120m | {fps:.1f} FPS", (panel_w + 15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)

    # Active Nudge / Control Banner (Right Panel Bottom)
    ctrl_col = (0, 255, 0) if "AUTOPILOT" in nudge_text else (0, 215, 255)
    cv2.rectangle(canvas, (panel_w + 15, panel_h - 40), (canvas_w - 15, panel_h - 10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (panel_w + 15, panel_h - 40), (canvas_w - 15, panel_h - 10), ctrl_col, 2)
    cv2.putText(canvas, f"TACTIC: {nudge_text}", (panel_w + 25, panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ctrl_col, 1)

    # Obstacle Hazard Banner (Left Panel Bottom)
    cv2.rectangle(canvas, (15, panel_h - 40), (panel_w - 15, panel_h - 10), (20, 20, 20), -1)
    cv2.rectangle(canvas, (15, panel_h - 40), (panel_w - 15, panel_h - 10), alert_color, 2)
    cv2.putText(canvas, status_text, (25, panel_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, alert_color, 1)

    # Traffic Signal HUD Badge (Center Top)
    cv2.rectangle(canvas, (canvas_w // 2 - 140, 8), (canvas_w // 2 + 140, 38), (20, 20, 20), -1)
    cv2.rectangle(canvas, (canvas_w // 2 - 140, 8), (canvas_w // 2 + 140, 38), tl_color, 2)
    cv2.circle(canvas, (canvas_w // 2 - 120, 23), 7, tl_color, -1)
    cv2.putText(canvas, tl_text, (canvas_w // 2 - 105, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, tl_color, 2)

    # Dedicated Bottom Legend Bar
    bar_y1 = panel_h
    bar_y2 = canvas_h
    cv2.rectangle(canvas, (0, bar_y1), (canvas_w, bar_y2), (15, 15, 15), -1)
    cv2.line(canvas, (0, bar_y1), (canvas_w, bar_y1), (60, 60, 60), 1)

    legend_items = [
        ("VEHICLE / 2W", COLOR_PALETTE[1]),
        ("JAYWALKER", COLOR_PALETTE[2]),
        ("STRAY / ANIMAL", COLOR_PALETTE[7]),
        ("POTHOLE", COLOR_PALETTE[6]),
        ("CURB / DIVIDER", COLOR_PALETTE[5]),
        ("DRIVABLE ROAD", COLOR_PALETTE[3]),
    ]
    
    col_w = canvas_w // len(legend_items)
    for idx, (label, color) in enumerate(legend_items):
        item_x = idx * col_w + 10
        item_y = bar_y1 + 28
        cv2.circle(canvas, (item_x, item_y - 4), 5, color, -1)
        cv2.putText(canvas, label, (item_x + 10, item_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (220, 220, 220), 1)

    return canvas

def main():
    global IS_RUNNING, WORLD_POTHOLE_LOCATIONS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Launching on: {torch.cuda.get_device_name(0)}")

    window_name = "CARLA Indian Road & Traffic Perception HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 900, 40)

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

    # Explicit local state tracking for autopilot
    is_autopilot_active = True
    vehicle.set_autopilot(True, traffic_manager.get_port())
    configure_traffic_manager_resilience(traffic_manager, vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(vehicle)

    # Stationary Potholes on Road Ahead
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
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "600000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "120")
    lidar_bp.set_attribute("upper_fov", "3.0")
    lidar_bp.set_attribute("lower_fov", "-25.0")

    lidar_transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=1.85))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(lidar)

    lidar_queue = queue.Queue(maxsize=5)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    print("[+] Indian Traffic Active: Reactive Swerve / Overtake Controller Running.")

    frame_counter = 0
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

            # Ego chassis point crop
            ego_mask = (xyz[:, 0] >= -2.2) & (xyz[:, 0] <= 2.2) & \
                       (xyz[:, 1] >= -1.0) & (xyz[:, 1] <= 1.0) & \
                       (xyz[:, 2] <= 0.2)
            xyz = xyz[~ego_mask]
            intensity = intensity[~ego_mask]

            # Road pothole injection
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

            # Dynamic pitch-invariant local plane fit
            fused_labels = extract_dynamic_elevation_features(xyz_valid, raw_preds)
            
            # Forward Corridor Threat Inspection
            status_text, alert_color, obstacle_info = inspect_forward_threats(xyz_valid, fused_labels)

            # Indian Traffic Reactive Nudge Controller (uses explicit boolean flag)
            nudge_text, is_autopilot_active = apply_reactive_nudge_control(
                vehicle, traffic_manager, obstacle_info, xyz_valid, is_autopilot_active
            )

            # Traffic Signal state
            tl_text, tl_color = detect_approaching_traffic_signal(world, vehicle, max_dist=25.0)

            fps = 1.0 / max(time.perf_counter() - t0, 1e-5)

            hud_image = render_dual_clipmap_hud(
                xyz_valid, fused_labels, status_text, alert_color, tl_text, tl_color, nudge_text, fps
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