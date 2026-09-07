import sys
import time
import queue
import cv2
import numpy as np
import torch
import carla

from models.native_backbone import NativeVoxelBackbone
from engine.clipmap_engine import FoveatedClipmapEngine

NUM_POINTS = 16384

class UltraFastCarlaPerception:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"[+] Initializing Perception on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # 1. Model Setup in pure Eager FP16 mode (avoids Windows Triton compilation crash)
        self.model = NativeVoxelBackbone(in_channels=4, num_classes=5).to(self.device).eval()
        ckpt = "./checkpoints/native_model_best.pth"
        try:
            self.model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
            print("[+] Checkpoint loaded successfully.")
        except Exception:
            print("[!] Checkpoint missing; running heuristic fallback.")

        self.clipmap_engine = FoveatedClipmapEngine(device=self.device)
        self.point_buffer = []

    @torch.inference_mode()
    def process_raw_bytes(self, raw_bytes):
        t0 = time.perf_counter()

        # 2. Transfer raw bytes directly to PyTorch GPU tensor without non-writable buffer warning
        flat_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
        points_gpu = flat_tensor.view(-1, 4).to(self.device, non_blocking=True).clone()
        
        # Coordinate conversion: Unreal (X-fwd, Y-right, Z-up) to ISO 8855 (X-fwd, Y-left, Z-up)
        points_gpu[:, 1] = -points_gpu[:, 1]

        # Accumulate up to 3 frames in VRAM
        self.point_buffer.append(points_gpu)
        if len(self.point_buffer) > 3:
            self.point_buffer.pop(0)

        accumulated_pts = torch.cat(self.point_buffer, dim=0)

        # 3. GPU-side distance partitioning and subsampling via torch.randperm
        xy_dist = torch.norm(accumulated_pts[:, :2], dim=1)
        near_indices = torch.nonzero(xy_dist <= 20.0).squeeze(-1)
        far_indices = torch.nonzero(xy_dist > 20.0).squeeze(-1)

        half_target = NUM_POINTS // 2
        selected = []

        if len(near_indices) > half_target:
            perm = torch.randperm(len(near_indices), device=self.device)[:half_target]
            selected.append(near_indices[perm])
        else:
            selected.append(near_indices)

        rem = NUM_POINTS - sum(len(x) for x in selected)
        if len(far_indices) > rem and rem > 0:
            perm = torch.randperm(len(far_indices), device=self.device)[:rem]
            selected.append(far_indices[perm])
        elif len(far_indices) > 0 and rem > 0:
            selected.append(far_indices[torch.randint(0, len(far_indices), (rem,), device=self.device)])

        sampled_scan = accumulated_pts[torch.cat(selected)]

        # 4. Pure FP16 Autocast Inference (Low latency on RTX 3050 Tensor Cores)
        model_in = sampled_scan.transpose(0, 1).unsqueeze(0)  # (1, 4, N)
        with torch.amp.autocast(self.device.type):
            logits = self.model(model_in)
            pred_classes = torch.argmax(logits[0], dim=0)

            # Heuristic assignment for surface extraction across full cloud
            full_classes = torch.zeros(accumulated_pts.shape[0], dtype=torch.long, device=self.device)
            full_classes[accumulated_pts[:, 2] < -1.2] = 1
            full_classes[accumulated_pts[:, 2] >= -1.0] = 3

            # Compute only Band 0 (Near: 20m) and Band 2 (Far: 120m)
            grids = self.clipmap_engine(accumulated_pts[:, :3], full_classes, active_bands=(0, 2))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        fps = 1.0 / max(time.perf_counter() - t0, 1e-5)
        return grids, fps


def main():
    # 1. Connect to CARLA
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 2. Synchronous Mode with 20 Hz Fixed Delta
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)

    blueprint_lib = world.get_blueprint_library()

    # 3. Spawn Ego Vehicle
    vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    vehicle.set_autopilot(True, traffic_manager.get_port())

    # 4. Spawn 120m Range 64-Channel LiDAR
    lidar_bp = blueprint_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("channels", "64")
    lidar_bp.set_attribute("points_per_second", "800000")
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("range", "120")
    lidar_bp.set_attribute("upper_fov", "15.0")
    lidar_bp.set_attribute("lower_fov", "-35.0")

    lidar_transform = carla.Transform(carla.Location(x=0.8, y=0.0, z=2.2))
    lidar_actor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    lidar_queue = queue.Queue()
    lidar_actor.listen(lambda data: lidar_queue.put(data))

    pipeline = UltraFastCarlaPerception()
    spectator = world.get_spectator()

    cv2.namedWindow("Foveated 2.5D Perception [Near 20m | Far 120m]", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Foveated 2.5D Perception [Near 20m | Far 120m]", 1000, 500)

    print("[+] System running locked at 20 Hz (delta=0.05s) with OpenCV renderer.")

    try:
        while True:
            # Advance simulation clock by 1 tick
            world.tick()

            # Retrieve frame from LiDAR queue
            try:
                lidar_data = lidar_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            # Update spectator camera behind ego car
            transform = vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=16) - transform.get_forward_vector() * 18,
                carla.Rotation(pitch=-35, yaw=transform.rotation.yaw)
            ))

            # Process frame
            grids, fps = pipeline.process_raw_bytes(lidar_data.raw_data)
            if grids is None:
                continue

            # Extract Band 0 (Near: 20m @ 10cm)
            near_z = grids[0][:, :, 1].cpu().numpy()
            near_step = grids[0][:, :, 3].cpu().numpy()

            # Extract Band 2 (Far: 120m @ 60cm)
            far_z = grids[2][:, :, 1].cpu().numpy()

            # Fast OpenCV Colormap Conversion
            near_norm = np.clip((near_z + 2.5) / 3.5 * 255.0, 0, 255).astype(np.uint8)
            near_bgr = cv2.applyColorMap(near_norm, cv2.COLORMAP_VIRIDIS)
            near_bgr[near_z == 0.0] = [20, 20, 20]

            # Highlight Curbs / Obstacles in Red
            curb_mask = (near_step > 0.12) & (near_z != 0.0)
            near_bgr[curb_mask] = [0, 0, 255]

            # Far Field Colormap
            far_norm = np.clip((far_z + 3.5) / 7.0 * 255.0, 0, 255).astype(np.uint8)
            far_bgr = cv2.applyColorMap(far_norm, cv2.COLORMAP_INFERNO)
            far_bgr[far_z == 0.0] = [20, 20, 20]

            # Vehicle markers
            cv2.circle(near_bgr, (200, 200), 4, (255, 255, 0), -1)
            cv2.circle(far_bgr, (200, 200), 3, (255, 255, 0), -1)

            # Orient forward
            near_bgr = cv2.flip(near_bgr, 0)
            far_bgr = cv2.flip(far_bgr, 0)

            # Combined canvas
            combined = np.hstack([near_bgr, far_bgr])

            # Text overlays
            cv2.putText(combined, "Near Field: 0-20m @ 10cm", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(combined, f"Far Horizon: 0-120m @ 60cm | {fps:.1f} FPS", (425, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            cv2.imshow("Foveated 2.5D Perception [Near 20m | Far 120m]", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[+] Exiting...")
    finally:
        print("[+] Restoring simulator settings and destroying actors...")
        world.apply_settings(original_settings)
        lidar_actor.stop()
        lidar_actor.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()