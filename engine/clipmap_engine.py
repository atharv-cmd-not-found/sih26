import torch
import torch.nn as nn

class FoveatedClipmapEngine(nn.Module):
    """
    Parallel GPU Scatter Engine projecting 3D point clouds into 
    concentric 2.5D elevation and obstacle layers with selective band rendering.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        # Concentric rings: [max_radius, resolution, grid_dimensions]
        self.bands = [
            {"max_r": 20.0,  "res": 0.10, "dim": 400},  # Near: 0-20m  @ 10cm (Index 0)
            {"max_r": 50.0,  "res": 0.25, "dim": 400},  # Mid:  20-50m @ 25cm (Index 1)
            {"max_r": 120.0, "res": 0.60, "dim": 400}   # Far:  0-120m @ 60cm (Index 2)
        ]

    @torch.no_grad()
    def forward(self, points, labels, active_bands=(0, 2)):
        """
        points: (N, 3) tensor [x, y, z] on GPU
        labels: (N,) tensor containing semantic class IDs (0 to 4) on GPU
        active_bands: tuple of band indices to compute (skips unrendered bands)
        Returns: dict mapping band_index -> (dim, dim, 5) tensor
                 Channels: [z_min, z_max, z_road, step_height, dominant_class]
        """
        grid_layers = {}

        for b_idx in active_bands:
            band = self.bands[b_idx]
            res, dim = band["res"], band["dim"]
            half = (dim * res) / 2.0
            num_cells = dim * dim

            # 1. GPU Bounding Box Filter
            mask = (points[:, 0].abs() < half) & (points[:, 1].abs() < half)
            b_pts = points[mask]
            b_lbl = labels[mask]

            if b_pts.shape[0] == 0:
                grid_layers[b_idx] = torch.zeros((dim, dim, 5), device=self.device)
                continue

            # 2. Discrete 2D index projection
            ix = ((b_pts[:, 0] + half) / res).long().clamp(0, dim - 1)
            iy = ((b_pts[:, 1] + half) / res).long().clamp(0, dim - 1)
            idx = iy * dim + ix
            z = b_pts[:, 2]

            # 3. Minimum and maximum elevation extraction
            z_min = torch.full((num_cells,), float("inf"), device=self.device)
            z_max = torch.full((num_cells,), float("-inf"), device=self.device)
            z_min = z_min.scatter_reduce(0, idx, z, reduce="amin", include_self=False)
            z_max = z_max.scatter_reduce(0, idx, z, reduce="amax", include_self=False)

            # 4. Road surface elevation (Class 1)
            road_mask = (b_lbl == 1)
            r_z = z[road_mask]
            r_idx = idx[road_mask]
            z_road = torch.zeros(num_cells, device=self.device)
            counts = torch.zeros(num_cells, device=self.device)
            if r_z.numel() > 0:
                z_road = z_road.scatter_add(0, r_idx, r_z)
                counts = counts.scatter_add(0, r_idx, torch.ones_like(r_z))
                valid = counts > 0
                z_road[valid] /= counts[valid]

            # 5. Semantic voting per cell
            one_hot = torch.nn.functional.one_hot(b_lbl.long(), num_classes=5).float()
            votes = torch.zeros((num_cells, 5), device=self.device)
            votes = votes.scatter_add(0, idx.unsqueeze(-1).expand(-1, 5), one_hot)
            dom_class = torch.argmax(votes, dim=-1)

            # 6. Step height / curb clearance calculation
            step_height = torch.where(torch.isinf(z_max - z_min), 0.0, z_max - z_min)
            z_min = torch.where(torch.isinf(z_min), 0.0, z_min)
            z_max = torch.where(torch.isinf(z_max), 0.0, z_max)

            layer = torch.stack([z_min, z_max, z_road, step_height, dom_class.float()], dim=-1)
            grid_layers[b_idx] = layer.reshape(dim, dim, 5)

        return grid_layers