#!/usr/bin/env python3
"""
Point Cloud Label Coherence Postprocessing Script

This script postprocesses a segmented point cloud (.ply file) to improve label coherence
by applying spatial clustering, noise filtering, and label propagation techniques.

Requirements:
    pip install open3d numpy scipy scikit-learn
"""

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import argparse
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class PointCloudPostprocessor:
    def __init__(self, epsilon=0.05, min_samples=10, k_neighbors=20):
        """
        Initialize the postprocessor with clustering parameters.
        
        Args:
            epsilon: DBSCAN epsilon parameter (neighborhood radius)
            min_samples: DBSCAN minimum samples per cluster
            k_neighbors: Number of neighbors for label propagation
        """
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.k_neighbors = k_neighbors
    
    def load_pointcloud(self, filepath):
        """Load point cloud from PLY file."""
        pcd = o3d.io.read_point_cloud(filepath)
        if len(pcd.points) == 0:
            raise ValueError(f"Could not load point cloud from {filepath}")
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Try to extract labels from colors (assuming labels are encoded in colors)
        labels = self._extract_labels_from_colors(colors) if colors is not None else None
        
        return points, colors, labels, pcd
    
    def _extract_labels_from_colors(self, colors):
        """
        Extract integer labels from RGB colors.
        Assumes labels are encoded as unique colors.
        """
        # Convert RGB to unique integers for labeling
        rgb_int = (colors * 255).astype(int)
        labels = rgb_int[:, 0] * 65536 + rgb_int[:, 1] * 256 + rgb_int[:, 2]
        
        # Map to consecutive integers starting from 0
        unique_labels = np.unique(labels)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[label] for label in labels])
        
        return mapped_labels
    
    def _labels_to_colors(self, labels):
        """Convert integer labels back to RGB colors for visualization."""
        num_labels = len(np.unique(labels))
        
        if HAS_MATPLOTLIB:
            colormap = plt.cm.get_cmap('tab20')
            colors = np.array([colormap(label / max(num_labels-1, 1))[:3] for label in labels])
        else:
            # Fallback: simple color mapping without matplotlib
            colors = np.zeros((len(labels), 3))
            for label in np.unique(labels):
                mask = labels == label
                colors[mask] = [
                    (label * 137) % 255 / 255.0,
                    (label * 199) % 255 / 255.0,
                    (label * 233) % 255 / 255.0
                ]
        
        return colors
    
    def remove_noise_points(self, points, labels, min_cluster_size=50):
        """Remove small clusters that are likely noise."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Identify noise labels (small clusters)
        noise_labels = unique_labels[counts < min_cluster_size]
        
        # Keep only points not in noise clusters
        valid_mask = ~np.isin(labels, noise_labels)
        
        print(f"Removed {len(noise_labels)} noise clusters with < {min_cluster_size} points")
        print(f"Kept {np.sum(valid_mask)} / {len(points)} points")
        
        return points[valid_mask], labels[valid_mask], valid_mask
    
    def spatial_label_smoothing(self, points, labels):
        """Apply spatial smoothing to make labels more coherent."""
        print("Applying spatial label smoothing...")
        
        # Build KNN index
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        smoothed_labels = labels.copy()
        
        for i in range(len(points)):
            # Get neighbors (excluding self)
            neighbor_indices = indices[i][1:]
            neighbor_labels = labels[neighbor_indices]
            
            # Vote for most common label among neighbors
            if len(neighbor_labels) > 0:
                most_common_label = stats.mode(neighbor_labels, keepdims=True)[0][0]
                
                # Only change label if there's strong consensus
                consensus_ratio = np.sum(neighbor_labels == most_common_label) / len(neighbor_labels)
                if consensus_ratio > 0.6:  # 60% consensus threshold
                    smoothed_labels[i] = most_common_label
        
        changes = np.sum(labels != smoothed_labels)
        print(f"Changed labels for {changes} points during smoothing")
        
        return smoothed_labels
    
    def refine_boundaries(self, points, labels):
        """Refine cluster boundaries using local spatial clustering."""
        print("Refining cluster boundaries...")
        
        refined_labels = labels.copy()
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == -1:  # Skip noise label
                continue
                
            label_mask = labels == label
            label_points = points[label_mask]
            
            if len(label_points) < self.min_samples:
                continue
            
            # Apply DBSCAN to points with this label
            clustering = DBSCAN(eps=self.epsilon, min_samples=self.min_samples)
            sub_labels = clustering.fit_predict(label_points)
            
            # If multiple clusters found, keep the largest one
            if len(np.unique(sub_labels)) > 1:
                unique_sub, counts = np.unique(sub_labels[sub_labels != -1], return_counts=True)
                if len(unique_sub) > 0:
                    main_cluster = unique_sub[np.argmax(counts)]
                    
                    # Mark points not in main cluster as noise
                    noise_mask = sub_labels != main_cluster
                    label_indices = np.where(label_mask)[0]
                    refined_labels[label_indices[noise_mask]] = -1
        
        return refined_labels
    
    def merge_small_clusters(self, points, labels, min_size=100):
        """Merge small clusters with their nearest larger neighbors."""
        print("Merging small clusters...")
        
        unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
        small_clusters = unique_labels[counts < min_size]
        
        merged_labels = labels.copy()
        
        for small_label in small_clusters:
            small_mask = labels == small_label
            small_points = points[small_mask]
            
            if len(small_points) == 0:
                continue
            
            # Find centroid of small cluster
            centroid = np.mean(small_points, axis=0)
            
            # Find nearest points from other clusters
            other_mask = (labels != small_label) & (labels != -1)
            if np.sum(other_mask) == 0:
                continue
                
            other_points = points[other_mask]
            other_labels = labels[other_mask]
            
            # Find closest point from another cluster
            distances = np.linalg.norm(other_points - centroid, axis=1)
            closest_idx = np.argmin(distances)
            target_label = other_labels[closest_idx]
            
            # Merge small cluster into target cluster
            merged_labels[small_mask] = target_label
            
        merged_count = len(small_clusters)
        print(f"Merged {merged_count} small clusters")
        
        return merged_labels
    
    def postprocess(self, points, labels):
        """Apply full postprocessing pipeline."""
        print(f"Starting postprocessing with {len(np.unique(labels))} initial labels")
        
        # Step 1: Remove noise points
        points_clean, labels_clean, valid_mask = self.remove_noise_points(points, labels)
        
        # Step 2: Spatial label smoothing
        labels_smooth = self.spatial_label_smoothing(points_clean, labels_clean)
        
        # Step 3: Refine boundaries
        labels_refined = self.refine_boundaries(points_clean, labels_smooth)
        
        # Step 4: Merge small clusters
        labels_final = self.merge_small_clusters(points_clean, labels_refined)
        
        print(f"Final result: {len(np.unique(labels_final[labels_final != -1]))} labels")
        
        return points_clean, labels_final, valid_mask
    
    def save_pointcloud(self, points, labels, output_path, original_pcd=None):
        """Save postprocessed point cloud to PLY file."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Generate colors from labels
        colors = self._labels_to_colors(labels)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Save to file
        success = o3d.io.write_point_cloud(output_path, pcd)
        if success:
            print(f"Saved postprocessed point cloud to {output_path}")
        else:
            print(f"Failed to save point cloud to {output_path}")
        
        return pcd


def main():
    parser = argparse.ArgumentParser(description="Postprocess segmented point cloud for label coherence")
    parser.add_argument("-i","--input_file", help="Input PLY file path")
    parser.add_argument("-o", "--output", help="Output PLY file path", default=None)
    parser.add_argument("--epsilon", type=float, default=0.05, help="DBSCAN epsilon parameter")
    parser.add_argument("--min_samples", type=int, default=10, help="DBSCAN min samples parameter")
    parser.add_argument("--k_neighbors", type=int, default=20, help="Number of neighbors for smoothing")
    parser.add_argument("--min_cluster_size", type=int, default=50, help="Minimum cluster size to keep")
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(args.input_file)[0]
        args.output = f"{base_name}_postprocessed.ply"
    
    # Initialize postprocessor
    postprocessor = PointCloudPostprocessor(
        epsilon=args.epsilon,
        min_samples=args.min_samples,
        k_neighbors=args.k_neighbors
    )
    
    try:
        # Load point cloud
        print(f"Loading point cloud from {args.input_file}")
        points, colors, labels, original_pcd = postprocessor.load_pointcloud(args.input_file)
        
        if labels is None:
            print("Warning: No labels found in point cloud. Creating dummy labels for demonstration.")
            # Create dummy labels for demonstration
            clustering = DBSCAN(eps=args.epsilon * 2, min_samples=args.min_samples)
            labels = clustering.fit_predict(points)
        
        # Postprocess
        points_processed, labels_processed, valid_mask = postprocessor.postprocess(points, labels)
        
        # Save result
        postprocessor.save_pointcloud(points_processed, labels_processed, args.output, original_pcd)
        
        print("\nPostprocessing completed successfully!")
        print(f"Input: {len(points)} points, {len(np.unique(labels))} labels")
        print(f"Output: {len(points_processed)} points, {len(np.unique(labels_processed[labels_processed != -1]))} labels")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())