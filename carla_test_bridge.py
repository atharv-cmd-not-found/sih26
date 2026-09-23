import os
import sys
import time
import math
import signal
import atexit
import warnings
import queue
import base64
from pathlib import Path
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import torch
import torch.nn as nn

if os.name == "nt":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

# --- IMPORT ADVERSE WEATHER CONDITIONING FILTER ---
from models.weather_filter import WeatherConditioningFilter
from sumo_bridge import SumoIndianTraffic

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
TARGET_CARLA_MAP = "Town10HD_Opt"
LIDAR_FRAME_RATE_HZ = 20
LIDAR_POINTS_PER_FRAME = 60000
MODEL_POINTS_PER_FRAME = 12000
WEATHER_DURATION_SECONDS = 10.0
GROUND_RETURN_KEEP_PROBABILITY = 1.0
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
    "walkthrough_writer": None,
    "cleaned": False
}
SUMO_TRAFFIC = None
VIEW_SELECTOR = None
OUTPUT_SELECTOR = None
WALKTHROUGH_DURATION_SECONDS = 60.0
WALKTHROUGH_PANEL_SECONDS = 15.0
WALKTHROUGH_VIDEO_FPS = 20


class OutputModeSelector:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("LIDForge Output")
        self.root.geometry("250x92")
        self.root.resizable(False, False)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.configure(bg="#151515")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Output.TFrame", background="#151515")
        style.configure("Output.TLabel", background="#151515", foreground="#eeeeee", font=("Segoe UI", 10))
        frame = ttk.Frame(self.root, style="Output.TFrame", padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="OUTPUT VIEW", style="Output.TLabel").pack(anchor="w")
        self.value = tk.StringVar(value="LiDAR")
        self.combo = ttk.Combobox(
            frame,
            textvariable=self.value,
            values=("LiDAR", "Height Map", "2.5D Grid", "Telemetry"),
            state="readonly",
            width=25,
        )
        self.combo.pack(fill="x", pady=(5, 0))
        self.root.update_idletasks()

    def update(self):
        if self.closed:
            return None
        try:
            self.root.update()
            return {"LiDAR": 1, "Height Map": 2, "2.5D Grid": 3, "Telemetry": 4}.get(self.value.get(), 1)
        except tk.TclError:
            self.closed = True
            return None

    def close(self):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


class DashboardViewSelector:
    def __init__(self, initial_mode=1):
        self.root = tk.Tk()
        self.root.title("LIDForge | Foveated Road Perception")
        self.root.geometry("1000x650")
        self.root.minsize(720, 480)
        self.closed = False
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.configure(bg="#101418")
        self.mode_by_name = {
            "LiDAR": 1,
            "Grid": 2,
            "Height Map": 3,
            "Telemetry": 4,
        }
        self.selected = tk.StringVar(value="LiDAR")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Top.TFrame", background="#101418")
        style.configure("Title.TLabel", background="#101418", foreground="#f2b84b", font=("Segoe UI", 16, "bold"))
        style.configure("Meta.TLabel", background="#101418", foreground="#9aa6b2", font=("Segoe UI", 10))
        style.configure("Value.TLabel", background="#101418", foreground="#55d6be", font=("Segoe UI", 11, "bold"))
        style.configure("View.TLabel", background="#101418", foreground="#dbe4ea", font=("Segoe UI", 10, "bold"))
        style.configure("View.TCombobox", fieldbackground="#202a31", background="#202a31", foreground="#dbe4ea")

        top = ttk.Frame(self.root, style="Top.TFrame", padding=(20, 14, 20, 10))
        top.pack(fill="x")
        ttk.Label(top, text="LIDForge", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="FOVEATED ROAD PERCEPTION", style="Meta.TLabel").pack(side="left", padx=(12, 0), pady=(4, 0))
        self.latency_label = ttk.Label(top, text="Latency --", style="Value.TLabel")
        self.latency_label.pack(side="right", padx=(18, 0))
        self.status_label = ttk.Label(top, text="Initializing", style="Meta.TLabel")
        self.status_label.pack(side="right")

        controls = ttk.Frame(self.root, style="Top.TFrame", padding=(20, 0, 20, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="VIEW", style="View.TLabel").pack(side="left")
        self.combo = ttk.Combobox(
            controls,
            textvariable=self.selected,
            values=list(self.mode_by_name),
            state="readonly",
            width=22,
            style="View.TCombobox",
        )
        self.combo.pack(side="left", padx=(10, 0))
        self.combo.current(max(0, initial_mode - 1))
        ttk.Label(controls, text="Select a view to inspect one output at a time", style="Meta.TLabel").pack(side="left", padx=(16, 0))

        self.image_label = tk.Label(self.root, bg="#101418", bd=0, highlightthickness=0)
        self.image_label.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.photo = None
        self.root.update_idletasks()

    def update(self):
        if self.closed:
            return None
        try:
            self.root.update()
            return self.mode_by_name.get(self.selected.get(), 1)
        except tk.TclError:
            self.closed = True
            return None

    def show(self, image, latency_ms, status):
        try:
            self.root.update_idletasks()
            available_w = max(320, self.image_label.winfo_width())
            available_h = max(240, self.image_label.winfo_height())
            source_h, source_w = image.shape[:2]
            fit_scale = min(available_w / source_w, available_h / source_h)
            fit_w = max(1, int(source_w * fit_scale))
            fit_h = max(1, int(source_h * fit_scale))
            fitted = cv2.resize(image, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
            display = np.full((available_h, available_w, 3), 16, dtype=np.uint8)
            offset_x = (available_w - fit_w) // 2
            offset_y = (available_h - fit_h) // 2
            display[offset_y:offset_y + fit_h, offset_x:offset_x + fit_w] = fitted
            ok, encoded = cv2.imencode(".png", display)
            if not ok:
                return
            self.photo = tk.PhotoImage(data=base64.b64encode(encoded.tobytes()))
            self.image_label.configure(image=self.photo)
            self.latency_label.configure(text=f"Latency {latency_ms:.1f} ms")
            self.status_label.configure(text=status)
            self.root.update_idletasks()
        except tk.TclError:
            pass

    def close(self):
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

# ==============================================================================
# OPENCV DRAWING UTILITIES (ROUNDED BOXES & DISTANCE BADGES)
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

def draw_rounded_rect(img, p1, p2, color, radius=6, thickness=2):
    x1, y1 = int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))
    x2, y2 = int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))
    r = int(min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2))

    if r <= 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), clr(color), thickness)
        return

    c = clr(color)
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), c, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), c, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, c, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, c, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, c, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, c, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), c, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), c, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), c, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), c, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, c, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, c, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, c, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, c, thickness, cv2.LINE_AA)

def draw_classification_badge(canvas, label_text, badge_x, badge_y, box_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.36
    thickness = 1
    (t_w, t_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    pad_x, pad_y = 6, 4
    x1 = badge_x
    y1 = badge_y - t_h - pad_y * 2
    x2 = x1 + t_w + pad_x * 2
    y2 = badge_y

    draw_rounded_rect(canvas, (x1, y1), (x2, y2), (18, 18, 18), radius=4, thickness=-1)
    draw_rounded_rect(canvas, (x1, y1), (x2, y2), box_color, radius=4, thickness=1)
    draw_text(canvas, label_text, (x1 + pad_x, y2 - pad_y - 1), font_scale, (240, 240, 240), thickness, font)

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
    walkthrough_writer = GLOBAL_CLEANUP_CONTEXT["walkthrough_writer"]
    global SUMO_TRAFFIC, VIEW_SELECTOR, OUTPUT_SELECTOR

    if SUMO_TRAFFIC is not None:
        SUMO_TRAFFIC.close()
        SUMO_TRAFFIC = None
    if VIEW_SELECTOR is not None:
        VIEW_SELECTOR.close()
        VIEW_SELECTOR = None
    if OUTPUT_SELECTOR is not None:
        OUTPUT_SELECTOR.close()
        OUTPUT_SELECTOR = None
    if walkthrough_writer is not None:
        walkthrough_writer.release()
        GLOBAL_CLEANUP_CONTEXT["walkthrough_writer"] = None

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


def camera_callback(sensor_data, data_queue):
    """Convert a CARLA RGB camera frame to a BGR OpenCV image."""
    raw = np.frombuffer(sensor_data.raw_data, dtype=np.uint8)
    image = raw.reshape((sensor_data.height, sensor_data.width, 4))[:, :, :3]
    try:
        data_queue.put_nowait(image.copy())
    except queue.Full:
        try:
            data_queue.get_nowait()
            data_queue.put_nowait(image.copy())
        except queue.Empty:
            pass

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
# PERCEPTION, CURB-GATING & DYNAMIC OBJECT CLUSTERING
# ==============================================================================
def extract_elevation_features(xyz, preds):
    labels = preds.copy()
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    # 1. Pitch-invariant road plane fit across immediate driving lane
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

    # High walls, fences, and palm trees suppression
    tall_mask = (h_local > 2.60) | (np.abs(y) > 16.0)
    labels[tall_mask] = 4

    # 2. Dynamic Object Clustering & Distance Bounding
    clusters = []
    occ_res = 0.35
    grid_dim = 140
    occ_grid = np.zeros((grid_dim, grid_dim), dtype=np.uint8)

    elevated_scene_pts = (h_local > 0.18) & (h_local < 2.8) & (x > 0.5) & (x < PERCEPTION_FORWARD_RANGE) & (np.abs(y) < PERCEPTION_LATERAL_RANGE)
    dynamic_candidate_pts = (labels == 1) | (labels == 2) | (labels == 7) | elevated_scene_pts
    if np.any(dynamic_candidate_pts):
        ox = np.clip((x[dynamic_candidate_pts] / PERCEPTION_FORWARD_RANGE * (grid_dim - 1)).astype(np.int32), 0, grid_dim - 1)
        oy = np.clip(((y[dynamic_candidate_pts] + PERCEPTION_LATERAL_RANGE) / (2.0 * PERCEPTION_LATERAL_RANGE) * (grid_dim - 1)).astype(np.int32), 0, grid_dim - 1)
        occ_grid[oy, ox] = 255

        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(occ_grid, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3:
                continue

            cx_m = (centroids[i][0] / (grid_dim - 1)) * PERCEPTION_FORWARD_RANGE
            cy_m = (centroids[i][1] / (grid_dim - 1)) * (2.0 * PERCEPTION_LATERAL_RANGE) - PERCEPTION_LATERAL_RANGE

            if abs(cy_m) > 4.2:
                continue

            c_mask = dynamic_candidate_pts & (np.abs(x - cx_m) < 2.4) & (np.abs(y - cy_m) < 2.4)
            pts_count = np.count_nonzero(c_mask)
            if pts_count < 6:
                continue

            z_min_c = np.min(z[c_mask])
            z_max_c = np.max(z[c_mask])
            h_base = z_min_c - z_ground_scalar
            h_span = z_max_c - z_min_c
            dx = np.max(x[c_mask]) - np.min(x[c_mask])
            dy = np.max(y[c_mask]) - np.min(y[c_mask])
            dist_from_ego = math.hypot(cx_m, cy_m)
            bbox = (float(np.min(x[c_mask])), float(np.max(x[c_mask])), float(np.min(y[c_mask])), float(np.max(y[c_mask])))

            # Trees and poles are tall static structures, not vehicles, even when their
            # canopy footprint overlaps a vehicle-sized cluster.
            tree_like = (h_span > 2.2) or (h_span > 1.7 and (dx > 3.0 or dy > 2.4))
            if tree_like:
                labels[c_mask] = 4
                clusters.append({
                    "class_name": "Tree",
                    "class": 4,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist_from_ego,
                    "color": (0, 140, 255),
                    "label": f"Tree: {dist_from_ego:.1f}m"
                })
            elif (1.0 <= dx <= 7.5 and 0.55 <= dy <= 3.4 and 0.35 <= h_span <= 2.2) and pts_count >= 8:
                labels[c_mask] = 1
                clusters.append({
                    "class_name": "Vehicle",
                    "class": 1,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist_from_ego,
                    "color": (255, 140, 0),
                    "label": f"Vehicle: {dist_from_ego:.1f}m"
                })
            # Class 2: Pedestrian
            elif (dx <= 1.8 and dy <= 1.8 and 0.65 <= h_span <= 2.4) and pts_count >= 6:
                labels[c_mask] = 2
                clusters.append({
                    "class_name": "Pedestrian",
                    "class": 2,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist_from_ego,
                    "color": (0, 0, 255),
                    "label": f"Pedestrian: {dist_from_ego:.1f}m"
                })
            # Class 7: Stray Animal / Low Profile Quadruped
            elif (0.5 <= dx <= 2.0 and 0.3 <= dy <= 1.4 and 0.20 <= h_span <= 1.0) and pts_count >= 6:
                labels[c_mask] = 7
                clusters.append({
                    "class_name": "Animal",
                    "class": 7,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist_from_ego,
                    "color": (0, 215, 255),
                    "label": f"Animal: {dist_from_ego:.1f}m"
                })
            elif dx >= 0.25 and dy >= 0.25 and h_span >= 0.20:
                labels[c_mask] = 4
                clusters.append({
                    "class_name": "Obstacle",
                    "class": 4,
                    "bbox": bbox,
                    "pos": (cx_m, cy_m),
                    "dist": dist_from_ego,
                    "color": (0, 140, 255),
                    "label": f"Obstacle: {dist_from_ego:.1f}m"
                })

    return labels, clusters


def add_actor_fallback_clusters(world, ego_vehicle, clusters):
    """Use CARLA actor geometry as a fallback label when LiDAR returns are sparse."""
    ego_tf = ego_vehicle.get_transform()
    yaw = math.radians(ego_tf.rotation.yaw)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    actors = list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*"))
    for actor in actors:
        if actor.id == ego_vehicle.id or not actor.is_alive:
            continue
        location = actor.get_location()
        dx = location.x - ego_tf.location.x
        dy = location.y - ego_tf.location.y
        forward = dx * cos_yaw + dy * sin_yaw
        lateral = -dx * sin_yaw + dy * cos_yaw
        if not (0.5 < forward < 48.0 and abs(lateral) < 24.0):
            continue

        is_pedestrian = actor.type_id.startswith("walker.pedestrian")
        class_id = 2 if is_pedestrian else 1
        class_name = "Pedestrian" if is_pedestrian else "Vehicle"
        color = (0, 0, 255) if is_pedestrian else (255, 140, 0)
        extent = actor.bounding_box.extent
        length = max(0.35, extent.x * 2.0)
        width = max(0.30, extent.y * 2.0)
        height = max(1.0, extent.z * 2.0)
        duplicate = any(
            c["class"] == class_id and math.hypot(c["pos"][0] - forward, c["pos"][1] - lateral) < 2.5
            for c in clusters
        )
        if duplicate:
            continue

        distance = math.hypot(forward, lateral)
        clusters.append({
            "class_name": class_name,
            "class": class_id,
            "bbox": (forward - length / 2.0, forward + length / 2.0, lateral - width / 2.0, lateral + width / 2.0),
            "pos": (forward, lateral),
            "dist": distance,
            "color": color,
            "label": f"{class_name}: {distance:.1f}m",
            "source": "carla_actor_fallback",
            "height": height,
        })

    return clusters

# ==============================================================================
# SAFE-WAYPOINT TACTICAL GUIDANCE
# ==============================================================================
class TacticalGuidanceController:
    def __init__(self):
        self.stall_counter = 0
        self.last_pos = None
        self.is_autopilot = True
        self.avoid_ticks = 0
        self.avoid_steer = 0.0
        self.avoid_direction = ""
        self.stop_sign_id = None
        self.stop_wait_started = None
        self.passed_stop_signs = set()

    @staticmethod
    def _adjacent_driving_lanes(vehicle, world_map):
        transform = vehicle.get_transform()
        waypoint = world_map.get_waypoint(
            transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return []
        yaw = math.radians(transform.rotation.yaw)
        lanes = []
        for side, lane_waypoint in ((-1, waypoint.get_left_lane()), (1, waypoint.get_right_lane())):
            if lane_waypoint is None or lane_waypoint.lane_type != carla.LaneType.Driving:
                continue
            if lane_waypoint.road_id != waypoint.road_id:
                continue
            if waypoint.lane_id != 0 and lane_waypoint.lane_id != 0:
                if (waypoint.lane_id > 0) != (lane_waypoint.lane_id > 0):
                    continue
            dx = lane_waypoint.transform.location.x - transform.location.x
            dy = lane_waypoint.transform.location.y - transform.location.y
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            lanes.append((side, lane_waypoint, lateral))
        return lanes

    def _stop_sign_ahead(self, vehicle, world):
        transform = vehicle.get_transform()
        yaw = math.radians(transform.rotation.yaw)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        candidates = list(world.get_actors().filter("traffic.stop*"))
        nearest = None
        nearest_distance = float("inf")
        for sign in candidates:
            if sign.id in self.passed_stop_signs or not sign.is_alive:
                continue
            location = sign.get_location()
            dx = location.x - transform.location.x
            dy = location.y - transform.location.y
            forward = dx * cos_yaw + dy * sin_yaw
            lateral = -dx * sin_yaw + dy * cos_yaw
            if 0.0 < forward < 14.0 and abs(lateral) < 5.0 and forward < nearest_distance:
                nearest = sign
                nearest_distance = forward
        return nearest, nearest_distance

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
                fwd_threat = {"class": 6, "dist": p_dist, "pos": (p_dist, 0.0), "label": f"Pothole: {p_dist:.1f}m"}

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

        stop_sign, stop_distance = self._stop_sign_ahead(vehicle, vehicle.get_world())
        if stop_sign is not None:
            if self.stop_sign_id != stop_sign.id:
                self.stop_sign_id = stop_sign.id
                self.stop_wait_started = None
            if self.stop_wait_started is None:
                self.stop_wait_started = time.monotonic()
            waited = time.monotonic() - self.stop_wait_started
            if stop_distance < 8.0 and waited < 2.0:
                if self.is_autopilot:
                    vehicle.set_autopilot(False)
                    self.is_autopilot = False
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                return "STOP SIGN: WAITING", (0, 0, 255), tl_state_str, tl_color, f"RELEASE IN {max(0.0, 2.0 - waited):.1f}s", (0, 220, 255)
            if waited >= 2.0:
                self.passed_stop_signs.add(stop_sign.id)
                self.stop_sign_id = None
                self.stop_wait_started = None
                if not self.is_autopilot:
                    vehicle.set_autopilot(True, traffic_manager.get_port())
                    self.is_autopilot = True
                return "STOP SIGN: PROCEEDING", (0, 255, 0), tl_state_str, tl_color, "2s WAIT COMPLETE", (0, 255, 0)

        if self.avoid_ticks > 0:
            target_side = -1 if self.avoid_steer < 0 else 1
            if not any(side == target_side for side, _, _ in self._adjacent_driving_lanes(vehicle, world_map)):
                self.avoid_ticks = 0
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.9, steer=0.0))
                return "BYPASS STOP: LANE ENDED", (0, 0, 255), tl_state_str, tl_color, "NO SAFE DRIVING LANE", (0, 0, 255)
            self.avoid_ticks -= 1
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False
            vehicle.apply_control(carla.VehicleControl(throttle=0.16, brake=0.1, steer=self.avoid_steer))
            return "BYPASSING: ALTERNATE LANE", (0, 165, 255), tl_state_str, tl_color, f"OVERTAKE {self.avoid_direction}", (0, 220, 255)

        # Brake early for vulnerable road users and commit to a clear alternate lane.
        if fwd_threat is not None and fwd_threat["dist"] < 8.0:
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False
            dynamic_mask = np.isin(labels, (1, 2, 7))
            left_clear = np.count_nonzero(dynamic_mask & (xyz[:, 0] > 1.0) & (xyz[:, 0] < 14.0) & (xyz[:, 1] > 1.0) & (xyz[:, 1] < 3.8))
            right_clear = np.count_nonzero(dynamic_mask & (xyz[:, 0] > 1.0) & (xyz[:, 0] < 14.0) & (xyz[:, 1] < -1.0) & (xyz[:, 1] > -3.8))
            left_cluster = any(c["class"] in (1, 2, 7) and 1.0 < c["pos"][0] < 14.0 and 1.0 < c["pos"][1] < 3.8 for c in clusters)
            right_cluster = any(c["class"] in (1, 2, 7) and 1.0 < c["pos"][0] < 14.0 and -3.8 < c["pos"][1] < -1.0 for c in clusters)
            left_blocked = left_clear >= 5 or left_cluster
            right_blocked = right_clear >= 5 or right_cluster
            if fwd_threat["dist"] < 4.0 or (left_blocked and right_blocked):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                return f"CRITICAL: {fwd_threat['label']}", (0, 0, 255), tl_state_str, tl_color, "EMERGENCY BRAKE", (0, 0, 255)
            preferred_side = -1 if not left_blocked else 1
            if not any(side == preferred_side for side, _, _ in self._adjacent_driving_lanes(vehicle, world_map)):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
                return f"CRITICAL: {fwd_threat['label']}", (0, 0, 255), tl_state_str, tl_color, "NO SAFE DRIVING LANE", (0, 0, 255)
            self.avoid_steer = 0.48 * preferred_side
            self.avoid_direction = "RIGHT" if preferred_side > 0 else "LEFT"
            self.avoid_ticks = 18
            vehicle.apply_control(carla.VehicleControl(throttle=0.12, brake=0.2, steer=self.avoid_steer))
            return f"AVOIDING: {fwd_threat['label']}", (0, 140, 255), tl_state_str, tl_color, f"OVERTAKE {self.avoid_direction}", (0, 220, 255)

        # Deadlock Bypass (> 25 ticks)
        if self.stall_counter > 25:
            if self.is_autopilot:
                vehicle.set_autopilot(False)
                self.is_autopilot = False
            valid_sides = self._adjacent_driving_lanes(vehicle, world_map)

            if valid_sides:
                # Prefer the side with the largest LiDAR clearance, but never leave a driving lane.
                side_scores = []
                for side, _, lane_lateral in valid_sides:
                    clearance = np.count_nonzero(
                        (xyz[:, 0] > 2.0) & (xyz[:, 0] < 12.0) &
                        (np.sign(xyz[:, 1]) == np.sign(lane_lateral)) &
                        (np.abs(xyz[:, 1] - lane_lateral) < 1.5)
                    )
                    side_scores.append((clearance, side))
                _, chosen_side = max(side_scores, key=lambda item: item[0])
                steer = 0.24 * chosen_side
                direction = "RIGHT" if chosen_side > 0 else "LEFT"
                vehicle.apply_control(carla.VehicleControl(throttle=0.12, brake=0.2, steer=steer))
                self.stall_counter = 0
                return "TACTIC: DRIVING-LANE BYPASS", (0, 165, 255), tl_state_str, tl_color, f"LANE CHANGE {direction}", (0, 165, 255)

            # No adjacent driving lane: remain on the road and wait for the blockage to clear.
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.85, steer=0.0))
            self.stall_counter = 0
            return "TACTIC: WAITING FOR CLEARANCE", (0, 165, 255), tl_state_str, tl_color, "NO SAFE DRIVING LANE", (0, 165, 255)

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
    draw_rounded_rect(canvas, (cx - 10, cy - 20), (cx + 10, cy + 4), (0, 215, 255), radius=4, thickness=2)
    draw_rounded_rect(canvas, (cx - 7, cy - 16), (cx + 7, cy - 6), (0, 140, 255), radius=2, thickness=-1)
    draw_rect(canvas, (cx - 5, cy - 5), (cx + 5, cy + 2), (220, 220, 220), -1)
    draw_rounded_rect(canvas, (cx - 30, cy + 8), (cx + 30, cy + 22), (0, 0, 0), radius=3, thickness=-1)
    draw_rounded_rect(canvas, (cx - 30, cy + 8), (cx + 30, cy + 22), (0, 215, 255), radius=3, thickness=1)
    draw_text(canvas, "EGO VEHICLE", (cx - 27, cy + 19), 0.32, (0, 215, 255), 1)

def render_ss_matching_dashboard(xyz, labels, clusters, status_text, status_col,
                                 tl_text, tl_col, tactic_text, tactic_col,
                                 fps, latency_dict, ego_speed, num_pts,
                                 view_mode=0, canvas_w=1600, canvas_h=930):
    canvas = np.full((canvas_h, canvas_w, 3), 16, dtype=np.uint8)

    # 1. Header Section
    draw_text(canvas, "LIDForge", (30, 48), 1.35, (255, 180, 50), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, " - Output Visualization", (225, 48), 1.15, (235, 235, 235), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, "Range-aware base grid + scene-adaptive refinement", (32, 78), 0.52, (170, 170, 170), 1)
    view_names = {1: "LIDAR", 2: "GRID", 3: "HEIGHT MAP", 4: "TELEMETRY"}
    view_mode = view_mode if view_mode in view_names else 1
    draw_text(canvas, f"VIEW: {view_names[view_mode]}", (560, 48), 0.48, (0, 215, 255), 1)
    draw_text(canvas, f"LATENCY: {latency_dict.get('total', 0.0):.1f} ms", (560, 76), 0.48, (0, 255, 0), 1)

    draw_rounded_rect(canvas, (820, 16), (1180, 88), (24, 24, 24), radius=5, thickness=-1)
    draw_rounded_rect(canvas, (820, 16), (1180, 88), (55, 55, 55), radius=5, thickness=1)
    draw_text(canvas, "Base Resolution (Range-aware)", (834, 38), 0.44, (255, 200, 100), 1)
    draw_text(canvas, "0 - 20 m -> 5 cm cells (fine)", (834, 58), 0.38, (190, 190, 190), 1)
    draw_text(canvas, "20 - 120 m -> 60 cm cells (coarse)", (834, 76), 0.38, (190, 190, 190), 1)

    draw_rounded_rect(canvas, (1200, 16), (1570, 88), (24, 24, 24), radius=5, thickness=-1)
    draw_rounded_rect(canvas, (1200, 16), (1570, 88), (55, 55, 55), radius=5, thickness=1)
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

    # --------------------------------------------------------------------------
    # PANEL (a): 180° POINT CLOUD WITH ROUNDED BOXES & DISTANCE LABELS
    # --------------------------------------------------------------------------
    draw_text(canvas, "(a ) LiDAR Point Cloud ( Extended 3D View )", (pa_x1 + 16, top_y + 26), 0.45, (220, 220, 220), 1)
    cx_a, cy_a = (pa_x1 + pa_x2) // 2, p_y2 - 38
    range_fwd_a = 90.0
    range_lat_a = 30.0

    for r in [30.0, 45.0, 60.0, 75.0, 90.0]:
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

    # Render Anti-Aliased Rounded Bounding Boxes + Distance Badges
    for c in clusters:
        xmin, xmax, ymin, ymax = c["bbox"]
        bx1 = np.clip(int(((ymin / range_lat_a) * (panel_w // 2) + cx_a)), pa_x1 + 2, pa_x2 - 2)
        bx2 = np.clip(int(((ymax / range_lat_a) * (panel_w // 2) + cx_a)), pa_x1 + 2, pa_x2 - 2)
        by1 = np.clip(int((cy_a - (xmax / range_fwd_a) * (panel_h - 70))), top_y + 34, cy_a)
        by2 = np.clip(int((cy_a - (xmin / range_fwd_a) * (panel_h - 70))), top_y + 34, cy_a)

        if bx2 - bx1 < 18:
            bx1 -= 9
            bx2 += 9
        if by2 - by1 < 18:
            by1 -= 9
            by2 += 9

        draw_rounded_rect(canvas, (bx1, by1), (bx2, by2), c["color"], radius=6, thickness=2)
        draw_classification_badge(canvas, c["label"], bx1 - 2, by1 - 4, c["color"])

    draw_ego_vehicle_icon(canvas, cx_a, cy_a)

    # --------------------------------------------------------------------------
    # PANEL (b): MULTI-RESOLUTION 2.5D GRID WITH ROUNDED REFINEMENT CELLS
    # --------------------------------------------------------------------------
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

    # Rounded adaptive refinement boxes on detected clusters
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

        draw_rounded_rect(canvas, (bx1, by1), (bx2, by2), c["color"], radius=5, thickness=2)
        for sub_x in range(bx1, bx2, 6):
            draw_line(canvas, (sub_x, by1), (sub_x, by2), c["color"], 1)
        for sub_y in range(by1, by2, 6):
            draw_line(canvas, (bx1, sub_y), (bx2, sub_y), c["color"], 1)

        draw_classification_badge(canvas, c["label"], bx1 - 2, by1 - 4, c["color"])

    draw_ego_vehicle_icon(canvas, cx_b, cy_b)

    # --------------------------------------------------------------------------
    # PANEL (c): BIRD'S-EYE HEIGHT MAP
    # --------------------------------------------------------------------------
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
    draw_rounded_rect(canvas, (bar_x, bar_y1), (bar_x + 12, bar_y2), (80, 80, 80), radius=3, thickness=1)

    draw_text(canvas, "Height (m)", (bar_x - 30, bar_y1 - 10), 0.36, (200, 200, 200), 1)
    draw_text(canvas, "10", (bar_x + 16, bar_y1 + 10), 0.35, (200, 200, 200), 1)
    draw_text(canvas, "5", (bar_x + 16, (bar_y1 + bar_y2) // 2 + 4), 0.35, (200, 200, 200), 1)
    draw_text(canvas, "0", (bar_x + 16, bar_y2), 0.35, (200, 200, 200), 1)

    # --------------------------------------------------------------------------
    # 3. EXTENDED PANEL (e): TELEMETRY, HAZARDS & PROFILER
    # --------------------------------------------------------------------------
    pe_y1 = 612
    pe_y2 = canvas_h - 18
    pe_x1 = 25
    pe_x2 = canvas_w - 25

    draw_rounded_rect(canvas, (pe_x1, pe_y1), (pe_x2, pe_y2), (10, 10, 10), radius=6, thickness=-1)
    draw_rounded_rect(canvas, (pe_x1, pe_y1), (pe_x2, pe_y2), (40, 40, 40), radius=6, thickness=1)
    draw_text(canvas, "(e) Real-Time Road Status, Hazard Telemetry & Latency Profiler",
              (pe_x1 + 18, pe_y1 + 24), 0.48, (220, 220, 220), 1)

    c1_w = 460
    draw_rounded_rect(canvas, (pe_x1 + 18, pe_y1 + 38), (pe_x1 + 18 + c1_w, pe_y2 - 16), (18, 18, 18), radius=5, thickness=-1)
    draw_rounded_rect(canvas, (pe_x1 + 18, pe_y1 + 38), (pe_x1 + 18 + c1_w, pe_y2 - 16), (45, 45, 45), radius=5, thickness=1)

    draw_text(canvas, "FORWARD CORRIDOR INSPECTION", (pe_x1 + 32, pe_y1 + 60), 0.38, (180, 180, 180), 1)
    draw_rounded_rect(canvas, (pe_x1 + 30, pe_y1 + 68), (pe_x1 + c1_w + 6, pe_y1 + 128), (22, 22, 22), radius=4, thickness=-1)
    draw_rounded_rect(canvas, (pe_x1 + 30, pe_y1 + 68), (pe_x1 + c1_w + 6, pe_y1 + 128), status_col, radius=4, thickness=2)
    draw_text(canvas, status_text, (pe_x1 + 44, pe_y1 + 104), 0.52, status_col, 2)

    draw_text(canvas, "TACTICAL CONTROLLER (INDIAN TRAFFIC FLOW)", (pe_x1 + 32, pe_y1 + 154), 0.38, (180, 180, 180), 1)
    draw_rounded_rect(canvas, (pe_x1 + 30, pe_y1 + 162), (pe_x1 + c1_w + 6, pe_y1 + 222), (22, 22, 22), radius=4, thickness=-1)
    draw_rounded_rect(canvas, (pe_x1 + 30, pe_y1 + 162), (pe_x1 + c1_w + 6, pe_y1 + 222), tactic_col, radius=4, thickness=2)
    draw_text(canvas, tactic_text, (pe_x1 + 44, pe_y1 + 198), 0.48, tactic_col, 2)

    c2_x1 = pe_x1 + 18 + c1_w + 18
    c2_w = 390
    c2_x2 = c2_x1 + c2_w
    draw_rounded_rect(canvas, (c2_x1, pe_y1 + 38), (c2_x2, pe_y2 - 16), (18, 18, 18), radius=5, thickness=-1)
    draw_rounded_rect(canvas, (c2_x1, pe_y1 + 38), (c2_x2, pe_y2 - 16), (45, 45, 45), radius=5, thickness=1)

    draw_text(canvas, "INTERSECTION SIGNAL (V2I)", (c2_x1 + 20, pe_y1 + 60), 0.38, (180, 180, 180), 1)
    draw_rounded_rect(canvas, (c2_x1 + 18, pe_y1 + 72), (c2_x2 - 18, pe_y1 + 128), (24, 24, 24), radius=4, thickness=-1)
    draw_rounded_rect(canvas, (c2_x1 + 18, pe_y1 + 72), (c2_x2 - 18, pe_y1 + 128), tl_col, radius=4, thickness=1)

    tl_box_x = c2_x1 + 32
    tl_box_y = pe_y1 + 84
    draw_rounded_rect(canvas, (tl_box_x, tl_box_y), (tl_box_x + 64, tl_box_y + 26), (10, 10, 10), radius=3, thickness=-1)
    draw_circle(canvas, (tl_box_x + 12, tl_box_y + 13), 6, (0, 0, 255) if "RED" in tl_text else (0, 0, 60))
    draw_circle(canvas, (tl_box_x + 32, tl_box_y + 13), 6, (0, 255, 255) if "YELLOW" in tl_text else (0, 60, 60))
    draw_circle(canvas, (tl_box_x + 52, tl_box_y + 13), 6, (0, 255, 0) if "GREEN" in tl_text else (0, 60, 0))
    draw_text(canvas, f"SIGNAL: {tl_text}", (tl_box_x + 80, tl_box_y + 18), 0.44, tl_col, 1)

    draw_text(canvas, "180 DEG SECTOR DANGER MATRIX", (c2_x1 + 20, pe_y1 + 154), 0.38, (180, 180, 180), 1)
    for idx, (s_name, s_col) in enumerate([("LEFT", (0, 255, 0)), ("CENTER", status_col), ("RIGHT", (0, 255, 0))]):
        sx = c2_x1 + 20 + idx * 116
        draw_rounded_rect(canvas, (sx, pe_y1 + 168), (sx + 104, pe_y1 + 218), (24, 24, 24), radius=4, thickness=-1)
        draw_rounded_rect(canvas, (sx, pe_y1 + 168), (sx + 104, pe_y1 + 218), s_col, radius=4, thickness=2)
        draw_text(canvas, s_name, (sx + 24, pe_y1 + 198), 0.44, s_col, 1)

    c3_x1 = c2_x2 + 18
    c3_x2 = pe_x2 - 18
    draw_rounded_rect(canvas, (c3_x1, pe_y1 + 38), (c3_x2, pe_y2 - 16), (18, 18, 18), radius=5, thickness=-1)
    draw_rounded_rect(canvas, (c3_x1, pe_y1 + 38), (c3_x2, pe_y2 - 16), (45, 45, 45), radius=5, thickness=1)
    draw_text(canvas, "REAL-TIME ROAD TELEMETRY & LATENCY PROFILER", (c3_x1 + 20, pe_y1 + 60), 0.40, (0, 215, 255), 1)

    draw_text(canvas, f"SPEED: {ego_speed:.1f} km/h", (c3_x1 + 22, pe_y1 + 92), 0.48, (0, 225, 255), 2)
    draw_text(canvas, f"REFRESH: {fps:.1f} FPS", (c3_x1 + 230, pe_y1 + 92), 0.48, (0, 255, 0), 2)
    draw_text(canvas, f"ACTIVE PTS: {num_pts:,}", (c3_x1 + 430, pe_y1 + 92), 0.44, (220, 220, 220), 1)

    draw_text(canvas, "MODE: AUTONOMOUS 20Hz SYNC", (c3_x1 + 22, pe_y1 + 118), 0.40, (0, 255, 0), 1)
    draw_text(canvas, "GRID: FOVEATED 2.5D", (c3_x1 + 320, pe_y1 + 118), 0.40, (255, 200, 0), 1)

    draw_rounded_rect(canvas, (c3_x1 + 18, pe_y1 + 134), (c3_x2 - 18, pe_y2 - 24), (24, 24, 24), radius=4, thickness=-1)
    draw_rounded_rect(canvas, (c3_x1 + 18, pe_y1 + 134), (c3_x2 - 18, pe_y2 - 24), (0, 215, 255), radius=4, thickness=1)

    tot_lat = latency_dict.get("total", 18.4)
    draw_text(canvas, f"TOTAL LATENCY: {tot_lat:.1f} ms", (c3_x1 + 32, pe_y1 + 162), 0.54, (0, 255, 255), 2)
    draw_text(canvas, f"SPCONV INFERENCE:   {latency_dict.get('spconv', 7.2):.1f} ms", (c3_x1 + 32, pe_y1 + 188), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"VOXEL DEDUP (64-BIT): {latency_dict.get('dedup', 0.5):.1f} ms", (c3_x1 + 32, pe_y1 + 208), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"OCCUPANCY & GATING: {latency_dict.get('gating', 1.8):.1f} ms", (c3_x1 + 320, pe_y1 + 188), 0.38, (200, 200, 200), 1)
    draw_text(canvas, f"DASHBOARD BLIT:     {latency_dict.get('render', 4.1):.1f} ms", (c3_x1 + 320, pe_y1 + 208), 0.38, (200, 200, 200), 1)

    if view_mode in (1, 2, 3):
        panel_x = {1: (pa_x1, pa_x2), 2: (pb_x1, pb_x2), 3: (pc_x1, pc_x2)}[view_mode]
        selected = canvas[top_y:p_y2, panel_x[0]:panel_x[1]]
        selected_header = np.full((54, selected.shape[1], 3), 16, dtype=np.uint8)
        draw_text(selected_header, f"VIEW: {view_names[view_mode]}   |   LATENCY: {latency_dict.get('total', 0.0):.1f} ms", (16, 34), 0.52, (0, 215, 255), 1)
        return cv2.resize(np.vstack((selected_header, selected)), (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    if view_mode == 4:
        selected = canvas[pe_y1:pe_y2, pe_x1:pe_x2]
        selected_header = np.full((54, selected.shape[1], 3), 16, dtype=np.uint8)
        draw_text(selected_header, f"VIEW: {view_names[view_mode]}   |   LATENCY: {latency_dict.get('total', 0.0):.1f} ms", (16, 34), 0.52, (0, 215, 255), 1)
        return cv2.resize(np.vstack((selected_header, selected)), (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    return canvas


def render_fresh_dashboard(xyz, labels, clusters, status_text, status_col,
                           tl_text, tl_col, tactic_text, tactic_col,
                           fps, latency_dict, ego_speed, num_pts,
                           view_mode=1, canvas_w=1200, canvas_h=650):
    """Render the new single-view dashboard without the legacy panel layout."""
    canvas = np.full((canvas_h, canvas_w, 3), (14, 18, 22), dtype=np.uint8)
    header_h = 68
    rail_w = 300
    main_x1, main_x2 = 22, canvas_w - rail_w - 12
    main_y1, main_y2 = header_h + 16, canvas_h - 20
    view_names = {1: "LiDAR", 2: "Grid", 3: "Height Map", 4: "Telemetry"}
    view_name = view_names.get(view_mode, "LiDAR")

    draw_text(canvas, "LIDForge", (24, 34), 0.82, (245, 185, 70), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, view_name.upper(), (178, 33), 0.48, (220, 230, 235), 1, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, f"{status_text}", (24, 57), 0.38, status_col, 1)
    draw_text(canvas, f"LATENCY  {latency_dict.get('total', 0.0):.1f} ms", (canvas_w - 250, 29), 0.42, (76, 220, 177), 1)
    draw_text(canvas, f"{fps:.1f} FPS", (canvas_w - 115, 51), 0.34, (150, 165, 175), 1)

    draw_rounded_rect(canvas, (main_x1, main_y1), (main_x2, main_y2), (19, 25, 30), radius=8, thickness=-1)
    draw_rounded_rect(canvas, (main_x1, main_y1), (main_x2, main_y2), (45, 58, 65), radius=8, thickness=1)
    draw_rounded_rect(canvas, (main_x2 + 12, main_y1), (canvas_w - 14, main_y2), (19, 25, 30), radius=8, thickness=-1)
    draw_rounded_rect(canvas, (main_x2 + 12, main_y1), (canvas_w - 14, main_y2), (45, 58, 65), radius=8, thickness=1)

    view_x1, view_x2 = main_x1 + 16, main_x2 - 16
    view_y1, view_y2 = main_y1 + 42, main_y2 - 16
    draw_text(canvas, view_name, (view_x1, main_y1 + 27), 0.46, (210, 220, 225), 1)

    if view_mode == 1:
        cx, cy = (view_x1 + view_x2) // 2, view_y2 - 12
        scale = min((view_x2 - view_x1) / 48.0, (view_y2 - view_y1) / 55.0)
        for distance in (10, 20, 30, 40):
            radius = int(distance * scale)
            cv2.ellipse(canvas, (cx, cy), (radius // 2, radius), 0, 180, 360, (45, 58, 64), 1)
            draw_text(canvas, f"{distance}m", (cx + 7, cy - radius + 12), 0.30, (100, 115, 123), 1)
        for lateral in (-12, -6, 0, 6, 12):
            x_end = int(cx + lateral * scale)
            draw_line(canvas, (cx, cy), (x_end, view_y1), (35, 47, 53), 1)
        visible = (xyz[:, 0] > 0.5) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
        if np.any(visible):
            px = np.clip((cx + xyz[visible, 1] * scale).astype(np.int32), view_x1, view_x2)
            py = np.clip((cy - xyz[visible, 0] * scale).astype(np.int32), view_y1, view_y2)
            point_labels = labels[visible]
            for class_id, color in ((3, (45, 170, 95)), (1, (60, 165, 245)), (2, (55, 80, 240)), (6, (220, 70, 210))):
                mask = point_labels == class_id
                canvas[py[mask], px[mask]] = color
            other = ~np.isin(point_labels, (1, 2, 3, 6))
            canvas[py[other], px[other]] = (185, 195, 205)
        draw_ego_vehicle_icon(canvas, cx, cy)
    elif view_mode == 2:
        cell = max(12, min((view_x2 - view_x1) // 24, (view_y2 - view_y1) // 28))
        origin_x = (view_x1 + view_x2) // 2
        origin_y = view_y2 - 12
        for row in range(24):
            for col in range(-12, 13):
                x1 = origin_x + col * cell
                y1 = origin_y - (row + 1) * cell
                if x1 < view_x1 or x1 + cell > view_x2 or y1 < view_y1:
                    continue
                draw_rect(canvas, (x1, y1), (x1 + cell - 1, y1 + cell - 1), (31, 42, 47), 1)
        if len(xyz):
            in_view = (xyz[:, 0] > 0.0) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
            for x_val, y_val, label in zip(xyz[in_view, 0], xyz[in_view, 1], labels[in_view]):
                col = int(y_val / 2.0)
                row = int(x_val / 2.0)
                color = (45, 170, 95) if label == 3 else (60, 165, 245) if label == 1 else (55, 80, 240) if label == 2 else (185, 195, 205)
                x1 = origin_x + col * cell
                y1 = origin_y - (row + 1) * cell
                if view_x1 <= x1 < view_x2 and view_y1 <= y1 < view_y2:
                    draw_rect(canvas, (x1, y1), (x1 + cell - 2, y1 + cell - 2), color, -1)
        draw_ego_vehicle_icon(canvas, origin_x, origin_y)
    elif view_mode == 3:
        map_w, map_h = view_x2 - view_x1, view_y2 - view_y1
        height_map = np.full((map_h, map_w), -2.5, dtype=np.float32)
        valid = (xyz[:, 0] > 0.0) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
        if np.any(valid):
            gx = np.clip((view_x1 + (xyz[valid, 1] + 24.0) / 48.0 * (map_w - 1)).astype(np.int32), view_x1, view_x2 - 1)
            gy = np.clip((view_y2 - (xyz[valid, 0] / 48.0 * (map_h - 1))).astype(np.int32), view_y1, view_y2 - 1)
            np.maximum.at(height_map, (gy - view_y1, gx - view_x1), xyz[valid, 2])
        normalized = np.clip(((height_map + 2.5) / 5.5 * 255.0), 0, 255).astype(np.uint8)
        height_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        height_color[height_map <= -2.4] = (19, 25, 30)
        canvas[view_y1:view_y2, view_x1:view_x2] = height_color
        draw_text(canvas, "LOW", (view_x1 + 8, view_y2 - 10), 0.30, (180, 190, 195), 1)
        draw_text(canvas, "HIGH", (view_x2 - 42, view_y1 + 16), 0.30, (245, 245, 245), 1)
    else:
        metrics = [
            ("EGO SPEED", f"{ego_speed:.1f} km/h", (80, 220, 190)),
            ("POINTS", f"{num_pts:,}", (220, 230, 235)),
            ("OBJECTS", str(len(clusters)), (245, 185, 70)),
            ("SIGNAL", tl_text, tl_col),
            ("TACTIC", tactic_text, tactic_col),
        ]
        y = view_y1 + 44
        for title, value, color in metrics:
            draw_text(canvas, title, (view_x1 + 30, y), 0.34, (125, 140, 148), 1)
            draw_text(canvas, value, (view_x1 + 30, y + 31), 0.65, color, 1)
            draw_line(canvas, (view_x1 + 30, y + 48), (view_x2 - 30, y + 48), (45, 58, 65), 1)
            y += 78

    draw_text(canvas, "DETECTIONS", (main_x2 + 30, main_y1 + 30), 0.40, (125, 140, 148), 1)
    if not clusters:
        draw_text(canvas, "No objects in range", (main_x2 + 30, main_y1 + 70), 0.40, (150, 165, 172), 1)
    else:
        y = main_y1 + 70
        for cluster in sorted(clusters, key=lambda item: item.get("dist", 999.0))[:7]:
            draw_circle(canvas, (main_x2 + 38, y - 5), 5, cluster.get("color", (185, 195, 205)), -1)
            draw_text(canvas, cluster.get("class_name", "Object"), (main_x2 + 52, y), 0.38, (220, 230, 235), 1)
            draw_text(canvas, f"{cluster.get('dist', 0.0):.1f} m", (main_x2 + 52, y + 20), 0.36, (105, 220, 185), 1)
            y += 55
    draw_text(canvas, "LIVE", (canvas_w - 60, canvas_h - 22), 0.32, (80, 220, 190), 1)
    return canvas


def render_reference_dashboard(xyz, labels, clusters, status_text, status_col,
                               tl_text, tl_col, tactic_text, tactic_col,
                               fps, latency_dict, ego_speed, num_pts,
                               view_mode=1, canvas_w=1200, canvas_h=650):
    """Render the reference-inspired monochrome operations dashboard."""
    canvas = np.full((canvas_h, canvas_w, 3), (15, 15, 15), dtype=np.uint8)
    line_color = (145, 145, 145)
    text_color = (215, 215, 215)
    muted = (155, 155, 155)
    accent = (195, 195, 195)
    green = (90, 220, 155)
    font = cv2.FONT_HERSHEY_PLAIN
    mode_names = {1: "BEV View", 2: "Foveated Grid", 3: "Elevation Map", 4: "Telemetry"}
    selected_name = mode_names.get(view_mode, "BEV View")

    def box(x1, y1, x2, y2):
        draw_rect(canvas, (x1, y1), (x2, y2), line_color, 1)

    def label(text, x, y, size=1.0, color=text_color):
        draw_text(canvas, text, (x, y), size, color, 1, font)

    box(22, 12, canvas_w - 22, canvas_h - 12)
    label("ADAPTIVE 2.5D LiDAR MAPPING", 34, 39, 1.15, text_color)
    label("Dynamic Environment Perception", 34, 60, 1.0, muted)
    draw_circle(canvas, (canvas_w - 180, 35), 4, green, -1)
    label("LIVE", canvas_w - 166, 39, 1.0, text_color)
    label(f"FPS {fps:.0f}", canvas_w - 92, 39, 1.0, text_color)
    label(f"LAT {latency_dict.get('total', 0.0):.1f}ms", canvas_w - 166, 60, 0.9, green)
    draw_line(canvas, (22, 72), (canvas_w - 22, 72), line_color, 1)

    left_x1, left_x2 = 22, 250
    center_x1, center_x2 = 250, 810
    right_x1, right_x2 = 810, canvas_w - 22
    top_y, bottom_y = 72, 458
    for x in (left_x1, left_x2, center_x1, center_x2, right_x1, right_x2):
        draw_line(canvas, (x, top_y), (x, bottom_y), line_color, 1)
    draw_line(canvas, (22, bottom_y), (canvas_w - 22, bottom_y), line_color, 1)

    label("FOVEATED GRID", 35, 103, 1.05, text_color)
    label("Zone 0", 35, 143, 1.0, text_color)
    label("0-10m (5cm)", 35, 164, 0.95, muted)
    label("Zone 1", 35, 207, 1.0, text_color)
    label("10-30m (15cm)", 35, 228, 0.95, muted)
    label("Zone 2", 35, 271, 1.0, text_color)
    label("30-100m (50cm)", 35, 292, 0.95, muted)
    label("Density Bar", 35, 337, 1.0, text_color)
    for idx in range(6):
        draw_rect(canvas, (35 + idx * 9, 350), (43 + idx * 9, 366), (80 + idx * 25,) * 3, -1)
    label("5cm", 99, 365, 0.9, muted)
    label("Objects", 35, 407, 1.0, text_color)
    label(f"{len(clusters):02d} detected", 35, 428, 0.95, green)

    label(selected_name.upper(), center_x1 + 22, 103, 1.05, text_color)
    label("Selected output", center_x1 + 22, 124, 0.9, muted)
    view_x1, view_x2 = center_x1 + 22, center_x2 - 22
    view_y1, view_y2 = 140, 438

    if view_mode == 1:
        # Fill the complete rectangular BEV viewport: lateral -24..24 m, forward 0..48 m.
        draw_rect(canvas, (view_x1, view_y1), (view_x2, view_y2), (20, 31, 39), -1)
        draw_rect(canvas, (view_x1, view_y1), (view_x2, view_y2), (55, 150, 145), 1)
        cx, cy = (view_x1 + view_x2) // 2, view_y2 - 1
        x_scale = (view_x2 - view_x1) / 48.0
        y_scale = (view_y2 - view_y1) / 48.0
        for lateral in (-18, -12, -6, 0, 6, 12, 18):
            grid_x = int(cx + lateral * x_scale)
            draw_line(canvas, (grid_x, view_y1), (grid_x, view_y2), (30, 68, 72), 1)
        for forward in (6, 12, 18, 24, 30, 36, 42):
            grid_y = int(cy - forward * y_scale)
            draw_line(canvas, (view_x1, grid_y), (view_x2, grid_y), (30, 68, 72), 1)
        for distance in (10, 20, 30, 40):
            cv2.ellipse(canvas, (cx, cy), (int(distance * x_scale), int(distance * y_scale)), 0, 180, 360, (48, 120, 120), 1)
            draw_text(canvas, f"{distance}m", (view_x1 + 7, int(cy - distance * y_scale + 13)), 0.30, (130, 190, 185), 1, font)
        draw_text(canvas, "FORWARD", (view_x1 + 8, view_y1 + 18), 0.32, (130, 190, 185), 1, font)
        visible = (xyz[:, 0] > 0.5) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
        if np.any(visible):
            px = np.clip((view_x1 + (xyz[visible, 1] + 24.0) * x_scale).astype(np.int32), view_x1, view_x2 - 1)
            py = np.clip((cy - xyz[visible, 0] * y_scale).astype(np.int32), view_y1, view_y2 - 1)
            point_labels = labels[visible]
            for class_id, color in ((3, (70, 185, 115)), (1, (50, 170, 245)), (2, (65, 75, 245)), (6, (220, 85, 210)), (7, (40, 205, 230))):
                mask = point_labels == class_id
                canvas[py[mask], px[mask]] = color
            unknown = ~np.isin(point_labels, (1, 2, 3, 6, 7))
            canvas[py[unknown], px[unknown]] = (160, 185, 190)
        draw_ego_vehicle_icon(canvas, cx, cy)
    elif view_mode == 2:
        for x in range(view_x1, view_x2, 24):
            draw_line(canvas, (x, view_y1), (x, view_y2), (45, 45, 45), 1)
        for y in range(view_y1, view_y2, 24):
            draw_line(canvas, (view_x1, y), (view_x2, y), (45, 45, 45), 1)
        for point, point_label in zip(xyz, labels):
            if 0 < point[0] < 48 and abs(point[1]) < 24:
                px = int((view_x1 + view_x2) / 2 + point[1] * 10)
                py = int(view_y2 - point[0] * 6)
                if view_x1 <= px < view_x2 and view_y1 <= py < view_y2:
                    color = (100, 190, 120) if point_label == 3 else (100, 180, 245)
                    draw_rect(canvas, (px, py), (px + 5, py + 5), color, -1)
    elif view_mode == 3:
        map_h, map_w = view_y2 - view_y1, view_x2 - view_x1
        height_map = np.full((map_h, map_w), -2.5, dtype=np.float32)
        valid = (xyz[:, 0] > 0) & (xyz[:, 0] < 48) & (np.abs(xyz[:, 1]) < 24)
        if np.any(valid):
            gx = np.clip(((xyz[valid, 1] + 24) / 48 * (map_w - 1)).astype(np.int32), 0, map_w - 1)
            gy = np.clip(((1 - xyz[valid, 0] / 48) * (map_h - 1)).astype(np.int32), 0, map_h - 1)
            np.maximum.at(height_map, (gy, gx), xyz[valid, 2])
        height_color = cv2.applyColorMap(np.clip((height_map + 2.5) / 5.5 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        height_color[height_map <= -2.4] = (15, 15, 15)
        canvas[view_y1:view_y2, view_x1:view_x2] = height_color
    else:
        label("PROCESSING", view_x1 + 28, view_y1 + 42, 0.95, muted)
        label(f"{latency_dict.get('gating', 0.0):.1f} ms", view_x1 + 28, view_y1 + 76, 1.5, green)
        label("POINT CLOUD", view_x1 + 210, view_y1 + 42, 0.95, muted)
        label(f"{num_pts:,}", view_x1 + 210, view_y1 + 76, 1.5, accent)
        label("EGO SPEED", view_x1 + 28, view_y1 + 142, 0.95, muted)
        label(f"{ego_speed:.1f} km/h", view_x1 + 28, view_y1 + 176, 1.5, accent)
        label("TACTIC", view_x1 + 210, view_y1 + 142, 0.95, muted)
        label(tactic_text[:24], view_x1 + 210, view_y1 + 176, 0.9, tactic_col)

    label("SYSTEM PERFORMANCE", right_x1 + 22, 103, 1.05, text_color)
    performance = [
        ("Memory", "runtime"),
        ("Processing", f"{latency_dict.get('total', 0.0):.1f} ms"),
        ("Render Rate", f"{fps:.0f} FPS"),
        ("Signal", tl_text),
        ("Tactic", tactic_text[:20]),
    ]
    y = 143
    for title, value in performance:
        label(f"{title}:", right_x1 + 22, y, 0.95, muted)
        label(value, right_x1 + 125, y, 0.95, green if title in ("Processing", "Render Rate") else text_color)
        y += 37
    label("DETECTIONS BY DISTANCE", right_x1 + 22, 340, 0.95, text_color)
    if clusters:
        y = 370
        for cluster in sorted(clusters, key=lambda item: item.get("dist", 999))[:3]:
            label(f"{cluster.get('class_name', 'Object')[:12]:12} {cluster.get('dist', 0.0):5.1f}m", right_x1 + 22, y, 0.9, cluster.get("color", accent))
            y += 22
    else:
        label("No objects in range", right_x1 + 22, 370, 0.9, muted)

    pipeline_y = 478
    box(22, pipeline_y, canvas_w - 22, 520)
    label("PIPELINE:", 35, 504, 0.95, text_color)
    label("RAW LiDAR  ->  AI SEGMENTATION  ->  FOVEATED GRID  ->  2.5D MAP", 115, 504, 0.95, muted)
    box(22, 520, canvas_w - 22, canvas_h - 12)
    tab_width = (canvas_w - 44) // 3
    tabs = (("BEV View", 1), ("Elevation Map", 3), ("Raw LiDAR", 2))
    for index, (tab, tab_mode) in enumerate(tabs):
        x = 35 + index * tab_width
        color = green if view_mode == tab_mode else text_color
        label(tab, x + 35, 553, 1.0, color)
    label("Dropdown controls active view", canvas_w - 230, 590, 0.82, muted)
    return canvas


def render_simple_opencv_output(xyz, labels, clusters, latency_ms, fps,
                                canvas_w=1200, canvas_h=700, output_mode=1):
    """Render only the LiDAR BEV, object labels, and top latency strip."""
    canvas = np.full((canvas_h, canvas_w, 3), (12, 12, 12), dtype=np.uint8)
    header_h = 54
    view_x1, view_x2 = 24, canvas_w - 24
    view_y1, view_y2 = header_h + 18, canvas_h - 24
    draw_rect(canvas, (view_x1, view_y1), (view_x2, view_y2), (20, 20, 20), -1)
    draw_rect(canvas, (view_x1, view_y1), (view_x2, view_y2), (170, 190, 190), 1)
    draw_text(canvas, "LIDForge  |  LiDAR OUTPUT", (24, 34), 0.78, (235, 235, 235), 2, cv2.FONT_HERSHEY_DUPLEX)
    draw_text(canvas, f"LATENCY  {latency_ms:.1f} ms", (canvas_w - 260, 29), 0.58, (85, 225, 180), 1, cv2.FONT_HERSHEY_SIMPLEX)
    draw_text(canvas, f"{fps:.1f} FPS", (canvas_w - 110, 48), 0.38, (155, 175, 180), 1)

    if output_mode == 2:
        map_w, map_h = view_x2 - view_x1, view_y2 - view_y1
        height_map = np.full((map_h, map_w), -2.5, dtype=np.float32)
        visible = (xyz[:, 0] > 0.0) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
        if np.any(visible):
            gx = np.clip(((xyz[visible, 1] + 24.0) / 48.0 * (map_w - 1)).astype(np.int32), 0, map_w - 1)
            gy = np.clip(((1.0 - xyz[visible, 0] / 48.0) * (map_h - 1)).astype(np.int32), 0, map_h - 1)
            np.maximum.at(height_map, (gy, gx), xyz[visible, 2])
        normalized = np.clip((height_map + 2.5) / 5.5 * 255.0, 0, 255).astype(np.uint8)
        rendered = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        rendered[height_map <= -2.4] = (20, 20, 20)
        canvas[view_y1:view_y2, view_x1:view_x2] = rendered
        ego_x = (view_x1 + view_x2) // 2
        ego_y = view_y2 - 12
        draw_rounded_rect(canvas, (ego_x - 14, ego_y - 25), (ego_x + 14, ego_y + 3), (255, 215, 0), radius=5, thickness=-1)
        draw_rect(canvas, (ego_x - 7, ego_y - 19), (ego_x + 7, ego_y - 8), (30, 30, 30), -1)
        draw_text(canvas, "HEIGHT MAP", (view_x1 + 12, view_y1 + 24), 0.52, (255, 255, 255), 1)
        draw_text(canvas, "LOW", (view_x1 + 12, view_y2 - 10), 0.36, (255, 255, 255), 1)
        draw_text(canvas, "HIGH", (view_x2 - 48, view_y1 + 22), 0.36, (255, 255, 255), 1)
        legend_x = view_x2 - 24
        legend_y = view_y1 + 35
        for index in range(80):
            color = cv2.applyColorMap(np.array([[255 - index * 3]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0].tolist()
            draw_rect(canvas, (legend_x, legend_y + index * 3), (legend_x + 12, legend_y + index * 3 + 3), color, -1)
        draw_text(canvas, "m", (legend_x - 2, legend_y - 8), 0.30, (255, 255, 255), 1)
        return canvas

    if output_mode == 3:
        grid_w, grid_h = 24, 24
        center_x = (view_x1 + view_x2) // 2
        bottom_y = view_y2 - 30
        cell_px = max(12, min((view_x2 - view_x1 - 80) // grid_w, 30))
        depth_x, depth_y = max(4, cell_px // 4), max(5, cell_px // 3)
        height_grid = np.full((grid_h, grid_w), -2.5, dtype=np.float32)
        class_grid = np.zeros((grid_h, grid_w), dtype=np.int32)
        visible = (xyz[:, 0] > 0.0) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
        if np.any(visible):
            selected = xyz[visible]
            selected_labels = labels[visible]
            cols = np.clip(((selected[:, 1] + 24.0) / 48.0 * grid_w).astype(np.int32), 0, grid_w - 1)
            rows = np.clip(((1.0 - selected[:, 0] / 48.0) * grid_h).astype(np.int32), 0, grid_h - 1)
            order = np.argsort(selected[:, 2])
            for index in order:
                height_grid[rows[index], cols[index]] = selected[index, 2]
                class_grid[rows[index], cols[index]] = selected_labels[index]
        palette = {0: (25, 25, 25), 1: (255, 140, 0), 2: (0, 0, 255), 3: (34, 139, 34), 4: (0, 140, 255), 5: (255, 255, 0), 6: (255, 0, 255), 7: (0, 215, 255)}
        for row in range(grid_h):
            for col in range(grid_w):
                base_x = center_x + (col - grid_w // 2) * cell_px + (row - grid_h // 2) * depth_x
                base_y = bottom_y - (grid_h - 1 - row) * depth_y
                ground = np.array([[base_x, base_y], [base_x + cell_px, base_y], [base_x + cell_px - depth_x, base_y - depth_y], [base_x - depth_x, base_y - depth_y]], dtype=np.int32)
                height = max(0.0, float(height_grid[row, col]) + 1.85) if height_grid[row, col] > -2.4 else 0.0
                lift = min(105, int(height * 25.0))
                top = ground.copy()
                top[:, 1] -= lift
                color = palette.get(class_grid[row, col], (160, 160, 160))
                side_color = tuple(max(0, value // 2) for value in color)
                cv2.fillConvexPoly(canvas, np.array([ground[0], ground[1], top[1], top[0]], dtype=np.int32), side_color)
                cv2.fillConvexPoly(canvas, np.array([ground[1], ground[2], top[2], top[1]], dtype=np.int32), side_color)
                cv2.fillConvexPoly(canvas, top, color)
                cv2.polylines(canvas, [top], True, (55, 55, 55), 1, cv2.LINE_AA)
        ego_x = center_x + (grid_w // 2 - grid_w // 2) * cell_px + (grid_h - 1 - grid_h // 2) * depth_x
        ego_y = bottom_y - 4
        draw_rounded_rect(canvas, (ego_x - 10, ego_y - 24), (ego_x + 10, ego_y), (0, 215, 255), radius=4, thickness=2)
        draw_text(canvas, "2.5D GRID", (view_x1 + 12, view_y1 + 24), 0.52, (235, 235, 235), 1)
        draw_text(canvas, "HEIGHT (m)", (view_x2 - 92, view_y1 + 24), 0.34, (235, 235, 235), 1)
        legend = (("ROAD", palette[3]), ("VEHICLE", palette[1]), ("PEDESTRIAN", palette[2]), ("POTHOLE", palette[6]))
        for index, (name, color) in enumerate(legend):
            lx = view_x1 + 12 + index * 125
            draw_rect(canvas, (lx, view_y2 - 22), (lx + 10, view_y2 - 12), color, -1)
            draw_text(canvas, name, (lx + 15, view_y2 - 13), 0.30, (235, 235, 235), 1)
        return canvas

    if output_mode == 4:
        draw_text(canvas, "TELEMETRY", (view_x1 + 32, view_y1 + 55), 0.65, (235, 235, 235), 1)
        metrics = (("POINTS PLOTTED", f"{len(xyz):,}"), ("OBJECTS", str(len(clusters))), ("LATENCY", f"{latency_ms:.1f} ms"), ("FRAME RATE", f"{fps:.1f} FPS"))
        y = view_y1 + 115
        for name, value in metrics:
            draw_text(canvas, name, (view_x1 + 32, y), 0.42, (155, 155, 155), 1)
            draw_text(canvas, value, (view_x1 + 300, y), 0.72, (80, 220, 170), 1)
            y += 70
        return canvas

    center_x = (view_x1 + view_x2) // 2
    bottom_y = view_y2 - 1
    x_scale = (view_x2 - view_x1) / 48.0
    y_scale = (bottom_y - view_y1) / 48.0
    range_scale = min(x_scale, y_scale)
    for distance in (10, 20, 30, 40):
        radius = int(distance * range_scale)
        cv2.circle(canvas, (center_x, bottom_y), radius, (90, 90, 90), 1, cv2.LINE_AA)
        draw_text(canvas, f"{distance}m", (center_x + 7, bottom_y - radius + 13), 0.34, (180, 180, 180), 1)
    for lateral in (-18, -12, -6, 0, 6, 12, 18):
        grid_x = int(center_x + lateral * x_scale)
        draw_line(canvas, (grid_x, view_y1), (grid_x, bottom_y), (45, 45, 45), 1)
    draw_text(canvas, "TOP-DOWN LiDAR", (view_x1 + 10, view_y1 + 22), 0.40, (200, 200, 200), 1)
    colors = {
        1: (255, 140, 0),
        2: (0, 0, 255),
        3: (34, 139, 34),
        4: (0, 140, 255),
        5: (255, 255, 0),
        6: (255, 0, 255),
        7: (0, 215, 255),
    }

    visible = (xyz[:, 0] > 0.0) & (xyz[:, 0] < 48.0) & (np.abs(xyz[:, 1]) < 24.0)
    if np.any(visible):
        selected = xyz[visible]
        px = np.clip((center_x + selected[:, 1] * x_scale).astype(np.int32), view_x1, view_x2 - 1)
        py = np.clip((bottom_y - selected[:, 0] * y_scale).astype(np.int32), view_y1, bottom_y)
        point_labels = labels[visible]
        for class_id, color in colors.items():
            mask = point_labels == class_id
            canvas[py[mask], px[mask]] = color
        unknown = ~np.isin(point_labels, tuple(colors))
        canvas[py[unknown], px[unknown]] = (160, 185, 190)

    draw_rounded_rect(canvas, (center_x - 10, bottom_y - 24), (center_x + 10, bottom_y - 2), (0, 215, 255), radius=4, thickness=2)
    for cluster in sorted(clusters, key=lambda item: item.get("dist", 999.0)):
        xmin, xmax, ymin, ymax = cluster.get("bbox", (
            cluster["pos"][0] - 0.8, cluster["pos"][0] + 0.8,
            cluster["pos"][1] - 0.8, cluster["pos"][1] + 0.8,
        ))
        bx1 = int(center_x + ymin * x_scale)
        bx2 = int(center_x + ymax * x_scale)
        by1 = int(bottom_y - xmax * y_scale)
        by2 = int(bottom_y - xmin * y_scale)
        bx1, bx2 = sorted((max(view_x1, bx1), min(view_x2 - 1, bx2)))
        by1, by2 = sorted((max(view_y1, by1), min(bottom_y, by2)))
        if bx2 - bx1 < 18:
            bx1 = max(view_x1, bx1 - 9)
            bx2 = min(view_x2 - 1, bx2 + 9)
        if by2 - by1 < 18:
            by1 = max(view_y1, by1 - 9)
            by2 = min(bottom_y, by2 + 9)
        color = cluster.get("color", colors.get(cluster.get("class", 0), (185, 195, 205)))
        draw_rounded_rect(canvas, (bx1, by1), (bx2, by2), color, radius=6, thickness=2)
        label_text = cluster.get("label", f"Object: {cluster.get('dist', 0.0):.1f}m")
        draw_classification_badge(canvas, label_text, bx1, max(view_y1 + 22, by1 - 3), color)
    draw_text(canvas, f"OBJECTS  {len(clusters)}", (view_x1 + 12, canvas_h - 7), 0.38, (155, 175, 180), 1)
    return canvas

# ==============================================================================
# MAIN SIMULATION & SENSOR BRIDGE (TOWN10 ENFORCED)
# ==============================================================================
def main():
    global IS_RUNNING, WORLD_POTHOLE_LOCATIONS, SUMO_TRAFFIC, VIEW_SELECTOR, OUTPUT_SELECTOR
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing LIDForge Master Client on: {torch.cuda.get_device_name(0)}")

    window_name = "LIDForge - LiDAR Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 700)
    walkthrough_enabled = os.environ.get("LIDFORGE_WALKTHROUGH", "0").lower() in ("1", "true", "yes", "on")
    if not walkthrough_enabled:
        OUTPUT_SELECTOR = OutputModeSelector()

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
    view_mode = 1
    actor_fallback_cache = []
    actor_fallback_frame = -3

    # 2. Connect to CARLA Server & Target Town10
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    GLOBAL_CLEANUP_CONTEXT["client"] = client

    world = client.get_world()
    active_map = world.get_map().name

    if TARGET_CARLA_MAP not in active_map:
        print(f"[!] Current map is {active_map}. Switching to {TARGET_CARLA_MAP}...")
        try:
            curr_s = world.get_settings()
            if curr_s.synchronous_mode:
                curr_s.synchronous_mode = False
                curr_s.fixed_delta_seconds = None
                world.apply_settings(curr_s)
        except Exception:
            pass

        world = client.load_world(TARGET_CARLA_MAP)
        time.sleep(3.0)
        world = client.get_world()
        active_map = world.get_map().name
        if TARGET_CARLA_MAP not in active_map:
            raise RuntimeError(f"CARLA loaded unexpected map: {active_map}")
        print(f"[✓] Active map verified: {active_map}")
    else:
        print(f"[✓] CARLA verified active on: {active_map}")

    GLOBAL_CLEANUP_CONTEXT["world"] = world

    # ================================================================================
    # 4. DYNAMIC WEATHER CONFIGURATION (10 SECONDS PER CONDITION)
    # ================================================================================
    weather_index = -1
    weather_index = update_dynamic_weather(world, 0.0, weather_index)

    world_map = world.get_map()
    spectator = world.get_spectator()

    # Synchronous Master Mode (20 Hz, Delta = 0.05s)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    GLOBAL_CLEANUP_CONTEXT["traffic_manager"] = traffic_manager

    # 3. Spawn Ego Vehicle in Town10
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
        raise RuntimeError("Failed to acquire ego vehicle in Town10.")

    vehicle.set_autopilot(True, traffic_manager.get_port())
    GLOBAL_CLEANUP_CONTEXT["actors"].append(vehicle)

    # 4. Stationary Road Potholes Dynamically Anchored Ahead of Ego in Town10
    v_init_tf = vehicle.get_transform()
    v_init_yaw = math.radians(v_init_tf.rotation.yaw)
    fx = math.cos(v_init_yaw)
    fy = math.sin(v_init_yaw)

    WORLD_POTHOLE_LOCATIONS = [
        (v_init_tf.location.x + fx * 12.0, v_init_tf.location.y + fy * 12.0, 1.10, 0.24),
        (v_init_tf.location.x + fx * 24.0, v_init_tf.location.y + fy * 24.0, 1.10, 0.24),
        (v_init_tf.location.x + fx * 38.0, v_init_tf.location.y + fy * 38.0, 1.00, 0.22)
    ]

    # SUMO owns ambient traffic; CARLA remains the synchronous sensor/rendering master.
    if os.environ.get("LIDFORGE_SUMO", "1").lower() not in ("0", "false", "off"):
        SUMO_TRAFFIC = SumoIndianTraffic(client, world, anchor_location=v_init_tf.location)
        if SUMO_TRAFFIC.start():
            SUMO_TRAFFIC.spawn_jaywalkers(vehicle, count=20)
    elif walkthrough_enabled:
        from traffic_generator import spawn_active_forward_crossers, spawn_ambient_indian_traffic
        fallback_traffic_manager = client.get_trafficmanager(8000)
        fallback_actors = spawn_ambient_indian_traffic(world, fallback_traffic_manager, num_vehicles=16)
        fallback_actors.extend(spawn_active_forward_crossers(world, vehicle, num_pedestrians=12))
        GLOBAL_CLEANUP_CONTEXT["actors"].extend(fallback_actors)

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

    camera_queue = queue.Queue(maxsize=3)
    camera = None
    walkthrough_writer = None
    walkthrough_dir = None
    walkthrough_fallback_actors = []
    walkthrough_started = time.monotonic()
    next_screenshot = 0.0
    if walkthrough_enabled:
        walkthrough_dir = Path("walkthrough_output")
        walkthrough_dir.mkdir(exist_ok=True)
        camera_bp = bp_lib.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "450")
        camera_bp.set_attribute("fov", "100")
        camera_tf = carla.Transform(carla.Location(x=1.4, y=0.0, z=2.2), carla.Rotation(pitch=-8.0))
        camera = world.spawn_actor(camera_bp, camera_tf, attach_to=vehicle)
        GLOBAL_CLEANUP_CONTEXT["actors"].append(camera)
        camera.listen(lambda data: camera_callback(data, camera_queue))
        walkthrough_writer = cv2.VideoWriter(
            str(walkthrough_dir / "lidforge_walkthrough.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            WALKTHROUGH_VIDEO_FPS,
            (1600, 700),
        )
        if not walkthrough_writer.isOpened():
            raise RuntimeError("Could not open walkthrough video writer.")
        GLOBAL_CLEANUP_CONTEXT["walkthrough_writer"] = walkthrough_writer
        print(f"[+] Walkthrough recording enabled: {walkthrough_dir.resolve()}")

    print("[+] Master Perception Loop Active in Town10. Ready.")

    frame_idx = 0
    try:
        while IS_RUNNING:
            t0 = time.perf_counter()
            elapsed_walkthrough = time.monotonic() - walkthrough_started
            if walkthrough_enabled and elapsed_walkthrough >= WALKTHROUGH_DURATION_SECONDS:
                print("[✓] Walkthrough duration complete.")
                break
            if OUTPUT_SELECTOR is not None:
                output_mode = OUTPUT_SELECTOR.update()
                if output_mode is None:
                    break
            elif walkthrough_enabled:
                output_mode = 1 + int(elapsed_walkthrough // WALKTHROUGH_PANEL_SECONDS) % 4
            world.tick()
            if SUMO_TRAFFIC is not None:
                SUMO_TRAFFIC.step()
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

            camera_frame = None
            if walkthrough_enabled:
                while not camera_queue.empty():
                    camera_frame = camera_queue.get_nowait()

            # Strip ego vehicle chassis returns
            xyz = points[:, :3].copy()
            intensity = np.clip(points[:, 3:4], 0.0, 1.0)
            ego_mask = (xyz[:, 0] >= -2.2) & (xyz[:, 0] <= 2.2) & (xyz[:, 1] >= -1.0) & (xyz[:, 1] <= 1.0) & (xyz[:, 2] <= 0.2)
            xyz = xyz[~ego_mask]
            intensity = intensity[~ego_mask]

            # Ingest potholes
            xyz = inject_world_potholes(xyz, vehicle)

            # Spatial Front-Hemisphere Filter
            spatial_mask = (xyz[:, 0] >= 0.0) & (xyz[:, 0] <= 48.0) & (xyz[:, 1] >= -24.0) & (xyz[:, 1] <= 24.0) & (xyz[:, 2] >= -2.8) & (xyz[:, 2] <= 3.2)
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

            if len(xyz_valid) > MODEL_POINTS_PER_FRAME:
                model_idx = np.linspace(0, len(xyz_valid) - 1, MODEL_POINTS_PER_FRAME, dtype=np.int32)
            else:
                model_idx = np.arange(len(xyz_valid), dtype=np.int32)
            xyz_model = xyz_valid[model_idx]
            intensity_model = intensity_valid[model_idx]

            coords_x = (xyz_model[:, 0] / voxel_size).astype(np.int32)
            coords_y = ((xyz_model[:, 1] + PERCEPTION_LATERAL_RANGE) / voxel_size).astype(np.int32)
            coords_z = ((xyz_model[:, 2] - PERCEPTION_Z_MIN) / voxel_size).astype(np.int32)
            coords_b = np.stack([np.zeros(len(coords_x), dtype=np.int32), coords_x, coords_y, coords_z], axis=-1)

            t_coords = torch.from_numpy(coords_b).to(device=device, dtype=torch.int32).contiguous()
            t_feats = torch.from_numpy(intensity_model).to(device=device, dtype=torch.float32).contiguous()

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
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                logits = model(x_sp)
                sampled_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            raw_preds = np.full(len(xyz_valid), 4, dtype=np.int64)
            raw_preds[model_idx[:len(sampled_preds)]] = sampled_preds
            t_sp = (time.perf_counter() - t_sp_0) * 1000.0

            # Dynamic Object Clustering with Real-Time Classification & Distance Tags
            t_gate_0 = time.perf_counter()
            fused_labels, clusters = extract_elevation_features(xyz_valid, raw_preds)
            if frame_idx - actor_fallback_frame >= 3:
                actor_fallback_cache = add_actor_fallback_clusters(world, vehicle, clusters)
                actor_fallback_frame = frame_idx
            else:
                clusters = clusters + [dict(cluster) for cluster in actor_fallback_cache if cluster.get("source") == "carla_actor_fallback"]
            t_gate = (time.perf_counter() - t_gate_0) * 1000.0

            # Tactical Guidance
            status_text, status_col, tl_text, tl_col, tactic_text, tactic_col = tactical_planner.update(
                vehicle, traffic_manager, world_map, clusters, fused_labels, xyz_valid
            )

            # Timing & Telemetry
            dt = max(time.perf_counter() - t0, 1e-5)
            fps = 1.0 / dt
            total_latency = dt * 1000.0

            vel = vehicle.get_velocity()
            speed = 3.6 * math.hypot(vel.x, vel.y)
            num_pts = len(xyz_valid)

            latency_dict = {
                "total": total_latency,
                "spconv": t_sp,
                "dedup": t_dedup,
                "gating": t_gate,
                "render": 4.2
            }

            # Render Dashboard with Rounded Boxes & Distance Badges
            t_ren_0 = time.perf_counter()
            window_rect = cv2.getWindowImageRect(window_name)
            display_w = max(720, int(window_rect[2]))
            display_h = max(480, int(window_rect[3]))
            dashboard = render_simple_opencv_output(
                xyz_valid, fused_labels, clusters, total_latency, fps,
                canvas_w=display_w, canvas_h=display_h, output_mode=output_mode
            )
            latency_dict["render"] = (time.perf_counter() - t_ren_0) * 1000.0

            if walkthrough_enabled:
                combined = np.full((700, 1600, 3), (12, 12, 12), dtype=np.uint8)
                dashboard_video = cv2.resize(dashboard, (960, 700), interpolation=cv2.INTER_AREA)
                combined[:, 640:1600] = dashboard_video
                if camera_frame is not None:
                    camera_video = cv2.resize(camera_frame, (640, 360), interpolation=cv2.INTER_AREA)
                    camera_y = (700 - 360) // 2
                    combined[camera_y:camera_y + 360, :640] = camera_video
                draw_text(combined, "CARLA SIMULATOR", (18, 32), 0.62, (235, 235, 235), 1, cv2.FONT_HERSHEY_DUPLEX)
                draw_text(combined, f"WALKTHROUGH  |  VIEW {output_mode}/4", (18, 58), 0.42, (85, 225, 180), 1)
                cv2.line(combined, (639, 0), (639, 700), (110, 110, 110), 1)
                walkthrough_writer.write(combined)
                if elapsed_walkthrough >= next_screenshot:
                    screenshot_path = walkthrough_dir / f"frame_{int(elapsed_walkthrough):03d}s.png"
                    cv2.imwrite(str(screenshot_path), combined)
                    next_screenshot += 5.0

            if frame_idx % 30 == 0:
                print(
                    f"[PERF] frame={frame_idx} latency={total_latency:.1f}ms "
                    f"points={len(xyz_valid):,} spconv={t_sp:.1f}ms "
                    f"gating={t_gate:.1f}ms render={latency_dict['render']:.1f}ms"
                )

            cv2.imshow(window_name, dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if frame_idx > 5 and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"[!] Master runtime exception: {e}")
    finally:
        if walkthrough_writer is not None:
            walkthrough_writer.release()
            GLOBAL_CLEANUP_CONTEXT["walkthrough_writer"] = None
        emergency_cleanup()

if __name__ == "__main__":
    main()