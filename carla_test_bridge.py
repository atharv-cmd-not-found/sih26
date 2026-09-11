import os
import sys
import time
import math
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
# 0: Background/Noise (Dark Slate Gray)
# 1: Vehicles (Bright Dodger Blue)
# 2: Pedestrians/Bicycles (Vivid Red)
# 3: Drivable Road (Forest Green)
# 4: Static Obstacles/Barriers (Bright Amber/Orange)
# 5: Curbs & Median Dividers (Cyan)
COLOR_PALETTE = {
    0: (40, 40, 40),
    1: (255, 120, 0),
    2: (0, 0, 255),
    3: (34, 139, 34),
    4: (0, 140, 255),
    5: (255, 255, 0),
}

def lidar_callback(sensor_data, data_queue):
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

def extract_curbs_and_obstacles(xyz, preds, z_ground_ref=-1.85):
    """
    Fuses deep semantic predictions with geometric step-height offsets
    to sharply classify road, curbs/dividers, and vehicle bodies.
    """
    labels = preds.copy()
    h_above_ground = xyz[:, 2] - z_ground_ref

    # 1. Flat Road Surface (-12 cm to +6 cm around ground)
    road_mask = (h_above_ground >= -0.12) & (h_above_ground < 0.06)
    labels[road_mask] = 3

    # 2. Curbs, Sidewalk Edges, and Median Dividers (6 cm to 45 cm above ground)
    divider_mask = (h_above_ground >= 0.06) & (h_above_ground <= 0.45)
    labels[divider_mask] = 5

    # 3. Vehicles and Static Obstacles (45 cm to 2.3 m above ground)
    raised_mask = (h_above_ground > 0.45) & (h_above_ground <= 2.30)
    # Retain dynamic class if the model recognized a vehicle/pedestrian
    labels[raised_mask & (labels == 1)] = 1
    labels[raised_mask & (labels == 2)] = 2
    labels[raised_mask & (labels != 1) & (labels != 2)] = 4

    return labels

def render_bev_hud(points, labels, canvas_size=800, range_m=40.0):
    hud = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    center = canvas_size // 2

    # Metric distance rings
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

    # Coordinate mapping: Forward X -> -Y (up on screen), Left Y -> -X (left on screen)
    px = ((y_val / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    py = (((-x_val) / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    px = np.clip(px, 0, canvas_size - 1)
    py = np.clip(py, 0, canvas_size - 1)

    # 1. Base Layer: Drivable Road (single-pixel splats)
    road_mask = (labels_val == 3)
    if np.any(road_mask):
        hud[py[road_mask], px[road_mask]] = COLOR_PALETTE[3]

    # 2. Curbs & Median Dividers (Cyan dilated edges)
    curb_mask = (labels_val == 5)
    if np.any(curb_mask):
        for rx, ry in zip(px[curb_mask], py[curb_mask]):
            cv2.circle(hud, (rx, ry), 2, COLOR_PALETTE[5], -1)

    # 3. Static Barriers & Walls (Orange)
    obs_mask = (labels_val == 4)
    if np.any(obs_mask):
        for rx, ry in zip(px[obs_mask], py[obs_mask]):
            cv2.circle(hud, (rx, ry), 2, COLOR_PALETTE[4], -1)

    # 4. Dynamic Objects: Vehicles (Blue) and Pedestrians (Red)
    for cls_idx, rad in [(1, 3), (2, 3)]:
        cls_mask = (labels_val == cls_idx)
        if np.any(cls_mask):
            c = COLOR_PALETTE[cls_idx]
            for rx, ry in zip(px[cls_mask], py[cls_mask]):
                cv2.circle(hud, (rx, ry), rad, c, -1)

    # Ego vehicle footprint
    cv2.rectangle(hud, (center - 6, center - 13), (center + 6, center + 13), (0, 255, 255), -1)
    cv2.line(hud, (center, center), (center, center - 17), (0, 0, 255), 2)

    # Legend Header
    cv2.rectangle(hud, (10, canvas_size - 38), (canvas_size - 10, canvas_size - 10), (18, 18, 18), -1)
    cv2.putText(hud, "BLUE: Car | CYAN: Divider/Curb | ORANGE: Barrier | GREEN: Road | RED: Ped",
                (18, canvas_size - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1)
    return hud

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Active Execution Device: {torch.cuda.get_device_name(0)}")

    window_name = "Calibrated SpConv 2.5D Perception HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Load Backbone
    model = SpConvUNet(in_channels=1, num_classes=5).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[+] SemanticKITTI checkpoint loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using geometric pipeline.")
    model.eval()

    # Simulator Connection
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()

    # Synchronous Execution Setup (20 Hz)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

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

    # CALIBRATED SENSOR SPECIFICATION
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "600000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "80")
    lidar_bp.set_attribute("upper_fov", "3.0")     # Suppresses sky noise, concentrates on curbs/cars
    lidar_bp.set_attribute("lower_fov", "-25.0")   # Eliminates bumper blind spot

    # Positioned at roof-front (x=0.8m) at standard height (z=1.85m)
    lidar_transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=1.85))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_queue = queue.Queue(maxsize=5)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    print("[+] Bridge active. Visualizing curbs, medians, and obstacles...")

    try:
        while True:
            world.tick()
            update_spectator_follow_cam(spectator, vehicle)

            # Flush queue to always evaluate the latest sweep
            points = None
            while not lidar_queue.empty():
                points = lidar_queue.get_nowait()

            if points is None:
                try:
                    points = lidar_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

            xyz = points[:, :3].copy()
            # Convert CARLA (UE4) to ISO 8855 right-handed frame (flip Y)
            xyz[:, 1] = -xyz[:, 1]
            intensity = np.clip(points[:, 3:4], 0.0, 1.0)

            # Voxelize to 0.05m resolution
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

            # Deduplicate indices to prevent SpConv rulebook collisions
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

            # Elevation-Aware Geometric Refinement (Sensor mounted at z=1.85m -> road plane at -1.85m)
            fused_labels = extract_curbs_and_obstacles(xyz_valid, raw_preds, z_ground_ref=-1.85)

            # Render BEV HUD
            hud_image = render_bev_hud(xyz_valid, fused_labels, canvas_size=800, range_m=40.0)
            cv2.imshow(window_name, hud_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("\n[+] Restoring simulator settings and cleaning actors...")
        try:
            traffic_manager.set_synchronous_mode(False)
            settings.synchronous_mode = False
            world.apply_settings(settings)
            lidar.stop()
            lidar.destroy()
            vehicle.destroy()
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()