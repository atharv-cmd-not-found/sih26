import os
import sys
import time
import math
import random
import signal
import atexit
import warnings
import queue
import traceback
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

# High-contrast BGR color palette
COLOR_PALETTE = {
    0: (40, 40, 40),     # Background / Clutter (Slate Gray)
    1: (255, 120, 0),    # Dynamic Vehicles (Dodger Blue)
    2: (0, 0, 255),      # Pedestrians / Bicycles (Vivid Red)
    3: (34, 139, 34),    # Drivable Road (Forest Green)
    4: (0, 140, 255),    # Barriers & Static Obstacles (Bright Amber)
    5: (255, 255, 0),    # Curbs & Median Dividers (Cyan)
}

# Global cleanup tracking state
IS_RUNNING = True
GLOBAL_CLEANUP_CONTEXT = {
    "world": None,
    "traffic_manager": None,
    "actors": [],
    "cleaned": False
}

def emergency_cleanup():
    """Guarantees CARLA resets to asynchronous mode and deletes all actors on exit."""
    if GLOBAL_CLEANUP_CONTEXT["cleaned"]:
        return
    GLOBAL_CLEANUP_CONTEXT["cleaned"] = True

    print("\n[+] Initiating graceful simulator release...")

    world = GLOBAL_CLEANUP_CONTEXT["world"]
    tm = GLOBAL_CLEANUP_CONTEXT["traffic_manager"]
    actors = GLOBAL_CLEANUP_CONTEXT["actors"]

    # 1. Reset CARLA to Asynchronous Mode FIRST so the engine unfreezes
    if world is not None:
        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("[✓] Restored CARLA to asynchronous mode.")
        except Exception as e:
            print(f"[!] Warning resetting world settings: {e}")

    # 2. Unsync Traffic Manager
    if tm is not None:
        try:
            tm.set_synchronous_mode(False)
            print("[✓] Unhooked TrafficManager synchronous mode.")
        except Exception:
            pass

    # 3. Destroy all spawned actors instantaneously via client batch commands
    if actors and world is not None:
        client = world.get_client()
        batch = []
        for actor in actors:
            if actor is not None and actor.is_alive:
                if hasattr(actor, 'stop'):
                    try:
                        actor.stop()
                    except Exception:
                        pass
                batch.append(carla.command.DestroyActor(actor))

        if batch:
            try:
                client.apply_batch_sync(batch, False)
                print(f"[✓] Successfully cleaned up {len(batch)} active actors (ego, NPCs, sensors).")
            except Exception as e:
                print(f"[!] Batch destruction warning: {e}")

    cv2.destroyAllWindows()
    print("[+] System shutdown complete. CARLA remains open and ready.")

def signal_handler(signum, frame):
    """Intercepts Ctrl+C or terminal termination signals."""
    global IS_RUNNING
    print("\n[!] Exit requested. Halting perception loop safely...")
    IS_RUNNING = False

# Register signal and process exit handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(emergency_cleanup)

def lidar_callback(sensor_data, data_queue):
    """Pushes raw point cloud into thread-safe queue."""
    raw_data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    data_queue.put(points)

def update_spectator_follow_cam(spectator, vehicle):
    """Smooth third-person chase camera locked to ego vehicle."""
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

def spawn_dynamic_actors(world, traffic_manager, num_vehicles=20, num_pedestrians=15):
    """Spawns autonomous vehicles and pedestrians around the map."""
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    spawned_actors = []

    # 1. Spawn Dynamic Vehicles
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicle_count = 0
    for sp in spawn_points:
        if vehicle_count >= num_vehicles:
            break
        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        npc = world.try_spawn_actor(bp, sp)
        if npc is not None:
            npc.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.vehicle_percentage_speed_difference(npc, random.uniform(10.0, 30.0))
            spawned_actors.append(npc)
            vehicle_count += 1

    print(f"[+] Spawned {vehicle_count} dynamic vehicles.")

    # 2. Spawn Dynamic Walkers
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    ped_count = 0
    for _ in range(num_pedestrians * 2):
        if ped_count >= num_pedestrians:
            break
        walker_bp = random.choice(walker_bps)
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            walker = world.try_spawn_actor(walker_bp, carla.Transform(loc))
            if walker is not None:
                controller_bp = bp_lib.find('controller.ai.walker')
                controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                controller.start()
                controller.go_to_location(world.get_random_location_from_navigation())
                controller.set_max_speed(1.4)
                spawned_actors.extend([controller, walker])
                ped_count += 1

    print(f"[+] Spawned {ped_count} pedestrians.")
    return spawned_actors

def extract_curbs_and_obstacles(xyz, preds, z_ground_ref=-1.85):
    """Fuses semantic class predictions with elevation-aware geometric filtering."""
    labels = preds.copy()
    h_above_ground = xyz[:, 2] - z_ground_ref

    # Drivable road bed (-12 cm to +6 cm)
    road_mask = (h_above_ground >= -0.12) & (h_above_ground < 0.06)
    labels[road_mask] = 3

    # Curbs, sidewalks, median dividers (+6 cm to +45 cm)
    divider_mask = (h_above_ground >= 0.06) & (h_above_ground <= 0.45)
    labels[divider_mask] = 5

    # Raised structures and bodies (> 45 cm)
    raised_mask = (h_above_ground > 0.45) & (h_above_ground <= 2.30)
    labels[raised_mask & (labels == 1)] = 1
    labels[raised_mask & (labels == 2)] = 2
    labels[raised_mask & (labels != 1) & (labels != 2)] = 4

    return labels

def render_bev_hud(points, labels, canvas_size=800, range_m=40.0):
    hud = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = canvas_size // 2

    # Distance concentric metric circles
    for dist in [5, 10, 20, 30]:
        radius_px = int((dist / range_m) * (canvas_size // 2))
        cv2.circle(hud, (center, center), radius_px, (45, 45, 45), 1)
        cv2.putText(hud, f"{dist}m", (center + 5, center - radius_px + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    cv2.line(hud, (center, 0), (center, canvas_size), (30, 30, 30), 1)
    cv2.line(hud, (0, center), (canvas_size, center), (30, 30, 30), 1)

    x = points[:, 0]
    y = points[:, 1]
    mask = (np.abs(x) < range_m) & (np.abs(y) < range_m)

    x_val = x[mask]
    y_val = y[mask]
    labels_val = labels[mask]

    # Synchronized coordinate projection (Vehicle frame: X forward, Y right)
    px = ((y_val / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    py = (((-x_val) / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    px = np.clip(px, 0, canvas_size - 1)
    py = np.clip(py, 0, canvas_size - 1)

    # 1. Road bed
    road_mask = (labels_val == 3)
    if np.any(road_mask):
        hud[py[road_mask], px[road_mask]] = COLOR_PALETTE[3]

    # 2. Curbs & Dividers
    curb_mask = (labels_val == 5)
    if np.any(curb_mask):
        for rx, ry in zip(px[curb_mask], py[curb_mask]):
            cv2.circle(hud, (rx, ry), 2, COLOR_PALETTE[5], -1)

    # 3. Barriers & Obstacles
    obs_mask = (labels_val == 4)
    if np.any(obs_mask):
        for rx, ry in zip(px[obs_mask], py[obs_mask]):
            cv2.circle(hud, (rx, ry), 2, COLOR_PALETTE[4], -1)

    # 4. Dynamic Vehicles & Pedestrians
    for cls_idx, rad in [(1, 3), (2, 3)]:
        cls_mask = (labels_val == cls_idx)
        if np.any(cls_mask):
            c = COLOR_PALETTE[cls_idx]
            for rx, ry in zip(px[cls_mask], py[cls_mask]):
                cv2.circle(hud, (rx, ry), rad, c, -1)

    # Ego vehicle footprint
    cv2.rectangle(hud, (center - 6, center - 13), (center + 6, center + 13), (0, 255, 255), -1)
    cv2.line(hud, (center, center), (center, center - 17), (0, 0, 255), 2)

    # HUD Legend Banner positioned on the right
    banner_w = 520
    banner_h = 30
    x1 = canvas_size - banner_w - 15
    y1 = canvas_size - banner_h - 15
    cv2.rectangle(hud, (x1, y1), (canvas_size - 15, canvas_size - 15), (18, 18, 18), -1)
    cv2.putText(hud, "BLUE: Car | RED: Ped | CYAN: Curb/Div | ORANGE: Barrier | GREEN: Road",
                (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)

    cv2.putText(hud, "Press 'Q', ESC or close [X] to Exit", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return hud

def main():
    global IS_RUNNING

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Execution Device: {torch.cuda.get_device_name(0)}")

    window_name = "Calibrated SpConv 2.5D Perception HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(window_name, 1050, 60)

    # Load Backbone
    model = SpConvUNet(in_channels=1, num_classes=5).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[+] Checkpoint loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using initialized backbone.")
    model.eval()

    # CARLA Client & World Configuration
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    spectator = world.get_spectator()
    GLOBAL_CLEANUP_CONTEXT["world"] = world

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    GLOBAL_CLEANUP_CONTEXT["traffic_manager"] = traffic_manager

    # Spawn Ego Vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    sp = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle is None:
        actors = world.get_actors().filter("vehicle.*")
        vehicle = actors[0] if len(actors) > 0 else None

    if vehicle is None:
        raise RuntimeError("Failed to obtain ego vehicle actor.")

    vehicle.set_autopilot(True, traffic_manager.get_port())
    GLOBAL_CLEANUP_CONTEXT["actors"].append(vehicle)

    # Spawn Dynamic NPCs & Walkers
    dynamic_actors = spawn_dynamic_actors(world, traffic_manager, num_vehicles=20, num_pedestrians=15)
    GLOBAL_CLEANUP_CONTEXT["actors"].extend(dynamic_actors)

    # Attach Calibrated LiDAR Sensor
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "600000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "80")
    lidar_bp.set_attribute("upper_fov", "3.0")
    lidar_bp.set_attribute("lower_fov", "-25.0")

    lidar_transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=1.85))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    GLOBAL_CLEANUP_CONTEXT["actors"].append(lidar)

    lidar_queue = queue.Queue(maxsize=5)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    print("[+] System active. Press 'q', 'ESC', close the window, or press Ctrl+C in terminal to stop.")

    try:
        while IS_RUNNING:
            world.tick()
            update_spectator_follow_cam(spectator, vehicle)

            # Flush queue to always inspect the latest sweep
            points = None
            while not lidar_queue.empty():
                points = lidar_queue.get_nowait()

            if points is None:
                try:
                    points = lidar_queue.get(timeout=0.05)
                except queue.Empty:
                    # Check window close even if sensor is idle
                    key = cv2.waitKey(1) & 0xFF
                    closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
                    if key in [ord('q'), ord('Q'), 27] or closed:
                        break
                    continue

            xyz = points[:, :3].copy()
            intensity = np.clip(points[:, 3:4], 0.0, 1.0)

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

            # Deduplicate voxel entries for rulebook stability
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

            # Sparse Convolution Inference
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    logits = model(x_sp)
                    raw_preds = torch.argmax(logits, dim=-1).cpu().numpy()

            # Geometric height fusion
            fused_labels = extract_curbs_and_obstacles(xyz_valid, raw_preds, z_ground_ref=-1.85)

            # Render HUD
            hud_image = render_bev_hud(xyz_valid, fused_labels, canvas_size=800, range_m=40.0)
            cv2.imshow(window_name, hud_image)

            # Check key presses AND window close button ("X")
            key = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1

            if key in [ord('q'), ord('Q'), 27] or window_closed:
                print("[+] HUD closed. Terminating loop cleanly...")
                break

    except Exception:
        traceback.print_exc()
    finally:
        emergency_cleanup()

if __name__ == "__main__":
    main()