import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.native_backbone import NativeVoxelBackbone
from engine.clipmap_engine import FoveatedClipmapEngine

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Initializing inference pipeline on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Load Pure PyTorch Model
    model = NativeVoxelBackbone(in_channels=4, num_classes=5).to(device).eval()
    ckpt_path = "./checkpoints/native_model_best.pth"

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[+] Successfully loaded weights from {ckpt_path}")
    else:
        print(f"[!] Warning: '{ckpt_path}' not found. Running with uninitialized weights.")

    clipmap_engine = FoveatedClipmapEngine(device=device)

    # Visualization setup
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#111")
    cmap = ListedColormap(["#111111", "#34495e", "#795548", "#e74c3c", "#2ecc71"])

    print("[+] Starting inference loop. Press Ctrl+C in terminal or close window to exit.")
    
    for frame in range(100):
        t0 = time.perf_counter()

        # Simulated LiDAR point batch: (1, 4, 16384) [x, y, z, intensity]
        pts_np = np.random.uniform(-40, 40, size=(16384, 3)).astype(np.float32)
        pts_np[:, 2] = np.random.normal(0.0, 0.4, size=(16384,))
        intensity = np.random.uniform(0, 1, size=(16384, 1)).astype(np.float32)
        scan = np.hstack([pts_np, intensity])

        scan_t = torch.from_numpy(scan).float().to(device)
        model_input = scan_t.transpose(0, 1).unsqueeze(0)  # Shape: (1, 4, N)

        # Forward Pass (Native FP16)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            logits = model(model_input)                     # Shape: (1, 5, N)
            pred_classes = torch.argmax(logits[0], dim=0)   # Shape: (N,)
            grids = clipmap_engine(scan_t[:, :3], pred_classes)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fps = 1.0 / max(time.perf_counter() - t0, 1e-5)

        # Plot Near-Field Band (0-10m @ 5cm)
        axes[0].clear()
        axes[0].set_facecolor("#111")
        axes[0].imshow(grids[0][:, :, 4].cpu().numpy(), cmap=cmap, extent=[-10, 10, -10, 10], vmin=0, vmax=4)
        axes[0].set_title("Near Field (0-10m @ 5cm)", color="white", fontsize=10)

        # Plot Far-Field Band (0-80m @ 40cm)
        axes[1].clear()
        axes[1].set_facecolor("#111")
        axes[1].imshow(grids[2][:, :, 4].cpu().numpy(), cmap=cmap, extent=[-80, 80, -80, 80], vmin=0, vmax=4)
        axes[1].set_title(f"Far Horizon (0-80m @ 40cm) | {fps:.1f} FPS", color="white", fontsize=10)

        plt.pause(0.01)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()