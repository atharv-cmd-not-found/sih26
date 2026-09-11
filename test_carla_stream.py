import os
import sys
import time
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

# Color map for 5 classes (BGR)
COLOR_MAP = {
    0: (100, 100, 100),  # Unlabeled
    1: (255, 0, 0),      # Vehicle (Blue)
    2: (0, 0, 255),      # Pedestrian (Red)
    3: (0, 200, 0),      # Drivable (Green)
    4: (0, 140, 255),    # Obstacle (Orange)
}

def lidar_callback(sensor_data, data_queue):
    raw_data = np.frombuffer(sensor_data.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(raw_data, (int(raw_data.shape[0] / 4), 4))
    data_queue.put(points)

def render_bev_hud(points, preds, canvas_size=700, range_m=40.0):
    hud = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    
    x = points[:, 0]
    y = points[:, 1]
    mask = (np.abs(x) < range_m) & (np.abs(y) < range_m)
    
    x_val = x[mask]
    y_val = y[mask]
    preds_val = preds[mask]
    
    px = ((y_val / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    py = (((-x_val) / range_m + 1.0) * 0.5 * (canvas_size - 1)).astype(np.int32)
    
    for cls_idx, color in COLOR_MAP.items():
        cls_mask = (preds_val == cls_idx)
        if np.any(cls_mask):
            hud[py[cls_mask], px[cls_mask]] = color
            
    center = canvas_size // 2
    cv2.circle(hud, (center, center), 5, (0, 255, 255), -1)
    cv2.line(hud, (center, center), (center, center - 15), (0, 255, 255), 2)
    cv2.putText(hud, "Ego Vehicle", (center + 8, center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(hud, "Blue: Vehicle | Red: Ped | Green: Road | Orange: Obstacle", 
                (10, canvas_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return hud

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running on GPU: {torch.cuda.get_device_name(0)}")

    # Load Model
    model = SpConvUNet(in_channels=1, num_classes=5).to(device)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
        print("[+] Checkpoint loaded successfully.")
    else:
        print(f"[!] Warning: Checkpoint missing at {CHECKPOINT_PATH}.")
    model.eval()

    # Connect to CARLA
    print("[+] Connecting to CARLA client...")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    sp = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], sp)
    if vehicle is None:
        actors = world.get_actors().filter("vehicle.*")
        vehicle = actors[0] if len(actors) > 0 else None
    
    if vehicle is None:
        raise RuntimeError("No vehicle available in CARLA.")
        
    vehicle.set_autopilot(True)
    print(f"[+] Ego vehicle active (ID: {vehicle.id})")

    # LiDAR configuration optimized for 6GB VRAM throughput
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "32")
    lidar_bp.set_attribute("points_per_second", "60000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "45")

    lidar = world.spawn_actor(lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.4)), attach_to=vehicle)
    lidar_queue = queue.Queue(maxsize=5)
    lidar.listen(lambda data: lidar_callback(data, lidar_queue))

    window_name = "CARLA SpConv Real-Time Perception HUD"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("[+] Sensor attached. Main perception loop active.")

    try:
        while True:
            world.tick()

            try:
                points = lidar_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            xyz = points[:, :3]
            intensity = points[:, 3:4]

            # Voxelize to 0.05m grid with boundary clipping
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

            b_indices = np.zeros((coords.shape[0], 1), dtype=np.int32)
            coords_b = np.hstack([b_indices, coords])

            t_coords = torch.from_numpy(coords_b).int().to(device)
            t_feats = torch.from_numpy(intensity).float().to(device)

            x_sp = spconv.SparseConvTensor(
                features=t_feats,
                indices=t_coords,
                spatial_shape=[3200, 3200, 160],
                batch_size=1
            )

            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    logits = model(x_sp)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()

            hud_image = render_bev_hud(xyz_valid, preds)
            cv2.imshow(window_name, hud_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception:
        print("\n[!] Error inside perception loop:")
        traceback.print_exc()

    finally:
        print("\n[+] Cleaning actors and restoring settings...")
        try:
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