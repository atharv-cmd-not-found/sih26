import numpy as np

class WeatherConditioningFilter:
    """
    De-noises LiDAR point clouds under degraded visual environments (rain, fog, dust).
    Filters out airborne volumetric backscatter and floating low-density ghost points.
    """
    def __init__(self, min_intensity=0.08, voxel_size=0.4, min_neighbors=3):
        self.min_intensity = min_intensity
        self.voxel_size = voxel_size
        self.min_neighbors = min_neighbors

    def filter_intensity(self, points, intensities):
        """Drops weak returns caused by water droplets and particulate matter."""
        valid_mask = intensities >= self.min_intensity
        return points[valid_mask], intensities[valid_mask]

    def remove_sparse_outliers(self, points, intensities):
        """Removes isolated floating returns using fast 3D spatial hashing."""
        if len(points) == 0:
            return points, intensities

        # Quantize points into discrete grid cells to count neighbors in O(N)
        grid_coords = np.floor(points[:, :3] / self.voxel_size).astype(np.int32)
        _, inverse_indices, counts = np.unique(
            grid_coords, axis=0, return_inverse=True, return_counts=True
        )

        cluster_counts = counts[inverse_indices]
        dense_mask = cluster_counts >= self.min_neighbors

        return points[dense_mask], intensities[dense_mask]

    def apply(self, point_cloud):
        """
        Processes Nx4 point cloud: [x, y, z, intensity]
        Returns: filtered Nx4 numpy array
        """
        if point_cloud.shape[1] < 4:
            pts, _ = self.remove_sparse_outliers(point_cloud[:, :3], None)
            return pts

        pts = point_cloud[:, :3]
        intensities = point_cloud[:, 3]

        # 1. Atmospheric backscatter filter
        pts_clean, intens_clean = self.filter_intensity(pts, intensities)

        # 2. Volumetric cluster filter
        pts_final, intens_final = self.remove_sparse_outliers(pts_clean, intens_clean)

        return np.hstack((pts_final, intens_final[:, np.newaxis]))