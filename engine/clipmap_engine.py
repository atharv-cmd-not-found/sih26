import torch
import torch.nn as nn
import math

class DynamicFoveatedGridEngine(nn.Module):
    """
    Multi-Resolution 2.5D Elevation & Semantic Grid Engine.
    - Base Near Grid: 0 to 20m @ 10cm cells (dim: 400x400)
    - Base Far Grid:  20 to 120m @ 60cm cells (dim: 400x400)
    - Dynamic Local Refinement: Dynamically instantiates high-res (5cm) 
      sub-patches for obstacles/curbs detected in the far field.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # Disjoint Base Range Configuration (No radial overlap)
        self.bands = [
            {"id": "near", "min_r": 0.0,  "max_r": 20.0,  "res": 0.10, "dim": 400},
            {"id": "far",  "min_r": 20.0, "max_r": 120.0, "res": 0.60, "dim": 400}
        ]
        
        # Dynamic Sub-Patch Configuration
        self.patch_res = 0.05    # 5 cm high-definition refinement
        self.patch_dim = 200    # 200x200 cells = 10.0m x 10.0m bounding ROI

    @torch.no_grad()
    def forward(self, points, labels, active_bands=(0, 1)):
        """
        points: (N, 3) tensor [x, y, z] on GPU
        labels: (N,) tensor of semantic classes (0=Unlabeled, 1=Road, 2=Terrain, 
                3=Static Obstacle, 4=Dynamic Vehicle, 5=Curb/Divider)
        Returns:
            base_grids: dict mapping band_id -> (dim, dim, 5) tensor
            refined_patch: dict containing dynamic sub-grid tensor and bounding extent, or None
        """
        base_grids = {}
        xy_dist = torch.norm(points[:, :2], dim=1)

        # 1. Rasterize Base Range Bands (Near and Far)
        for b_idx in active_bands:
            band = self.bands[b_idx]
            res = band["res"]
            dim = band["dim"]
            half = (dim * res) / 2.0
            num_cells = dim * dim

            # Spatial partition: radial envelope & square boundary
            mask = (xy_dist >= band["min_r"]) & (xy_dist < band["max_r"]) & \
                   (points[:, 0].abs() < half) & (points[:, 1].abs() < half)

            b_pts = points[mask]
            b_lbl = labels[mask]

            if b_pts.shape[0] == 0:
                base_grids[band["id"]] = torch.zeros((dim, dim, 5), device=self.device)
                continue

            # Project into discrete 2D cell indices
            # ix corresponds to forward X, iy corresponds to lateral Y
            ix = ((b_pts[:, 0] + half) / res).long().clamp(0, dim - 1)
            iy = ((b_pts[:, 1] + half) / res).long().clamp(0, dim - 1)
            linear_idx = iy * dim + ix
            z = b_pts[:, 2]

            # Minimum and Maximum surface elevation
            z_min = torch.full((num_cells,), float("inf"), device=self.device)
            z_max = torch.full((num_cells,), float("-inf"), device=self.device)
            z_min = z_min.scatter_reduce(0, linear_idx, z, reduce="amin", include_self=False)
            z_max = z_max.scatter_reduce(0, linear_idx, z, reduce="amax", include_self=False)

            # Road elevation plane (Class 1 = Drivable Road)
            road_mask = (b_lbl == 1)
            r_z = z[road_mask]
            r_idx = linear_idx[road_mask]
            z_road = torch.zeros(num_cells, device=self.device)
            counts = torch.zeros(num_cells, device=self.device)
            if r_z.numel() > 0:
                z_road = z_road.scatter_add(0, r_idx, r_z)
                counts = counts.scatter_add(0, r_idx, torch.ones_like(r_z))
                valid = counts > 0
                z_road[valid] /= counts[valid]

            # Semantic voting aggregation (6 classes: 0 to 5)
            one_hot = torch.nn.functional.one_hot(b_lbl.long(), num_classes=6).float()
            votes = torch.zeros((num_cells, 6), device=self.device)
            votes = votes.scatter_add(0, linear_idx.unsqueeze(-1).expand(-1, 6), one_hot)
            dom_class = torch.argmax(votes, dim=-1)

            # Structural step height (z_max - z_min)
            step_height = torch.where(torch.isinf(z_max - z_min), 0.0, z_max - z_min)
            z_min = torch.where(torch.isinf(z_min), 0.0, z_min)
            z_max = torch.where(torch.isinf(z_max), 0.0, z_max)

            # Channels: [z_min, z_max, z_road, step_height, dominant_class]
            layer = torch.stack([z_min, z_max, z_road, step_height, dom_class.float()], dim=-1)
            base_grids[band["id"]] = layer.reshape(dim, dim, 5)

        # 2. Scene-Driven Dynamic Local Refinement
        refined_patch = None
        # Trigger on dynamic vehicles (Class 4) or prominent curbs (Class 5) located beyond the near zone (r >= 20m)
        trigger_mask = (xy_dist >= 20.0) & ((labels == 4) | (labels == 5))
        
        if torch.any(trigger_mask):
            target_pts = points[trigger_mask]
            # Identify closest dynamic/geometric obstacle cluster centroid
            target_dists = torch.norm(target_pts[:, :2], dim=1)
            closest_idx = torch.argmin(target_dists)
            cx = target_pts[closest_idx, 0].item()
            cy = target_pts[closest_idx, 1].item()

            patch_half = (self.patch_dim * self.patch_res) / 2.0  # 5.0m half-extent

            # Crop local ROI window around the target
            roi_mask = (points[:, 0] >= (cx - patch_half)) & (points[:, 0] < (cx + patch_half)) & \
                       (points[:, 1] >= (cy - patch_half)) & (points[:, 1] < (cy + patch_half))

            p_pts = points[roi_mask]
            p_lbl = labels[roi_mask]

            if p_pts.shape[0] > 0:
                p_ix = ((p_pts[:, 0] - (cx - patch_half)) / self.patch_res).long().clamp(0, self.patch_dim - 1)
                p_iy = ((p_pts[:, 1] - (cy - patch_half)) / self.patch_res).long().clamp(0, self.patch_dim - 1)
                p_idx = p_iy * self.patch_dim + p_ix
                p_total = self.patch_dim * self.patch_dim

                p_z = p_pts[:, 2]
                p_z_max = torch.full((p_total,), float("-inf"), device=self.device)
                p_z_max = p_z_max.scatter_reduce(0, p_idx, p_z, reduce="amax", include_self=False)
                p_z_max = torch.where(torch.isinf(p_z_max), 0.0, p_z_max)

                p_one_hot = torch.nn.functional.one_hot(p_lbl.long(), num_classes=6).float()
                p_votes = torch.zeros((p_total, 6), device=self.device)
                p_votes = p_votes.scatter_add(0, p_idx.unsqueeze(-1).expand(-1, 6), p_one_hot)
                p_dom_class = torch.argmax(p_votes, dim=-1)

                patch_tensor = torch.stack([p_z_max, p_dom_class.float()], dim=-1).reshape(self.patch_dim, self.patch_dim, 2)
                refined_patch = {
                    "tensor": patch_tensor,
                    "center": (cx, cy),
                    "dist": math.sqrt(cx**2 + cy**2),
                    "res": self.patch_res
                }

        return base_grids, refined_patch