import numpy as np


def cluster_points_grid(points, eps):
    """
    Cluster a large set of 2D points using a grid-based approach with union-find.

    Args:
        points (np.ndarray): Array of shape (N,2) with point coordinates (x,y) as float64.
        eps (float): Maximum distance to merge clusters.

    Returns:
        list of tuple: A list of centroids (x, y) for each cluster.
    """
    cell_size = eps
    n = points.shape[0]
    cell_map = {}

    # 1. Partition points into grid cells.
    for idx in range(n):
        x, y = points[idx]
        cell = (int(x // cell_size), int(y // cell_size))
        if cell not in cell_map:
            cell_map[cell] = []
        cell_map[cell].append(idx)

    # 2. Each cell becomes an initial cluster.
    clusters = {}  # Mapping: cell -> cluster id
    cluster_points_dict = {}  # Mapping: cluster id -> list of point indices
    next_cluster_id = 0
    for cell, idxs in cell_map.items():
        clusters[cell] = next_cluster_id
        cluster_points_dict[next_cluster_id] = idxs.copy()
        next_cluster_id += 1

    # 3. Initialize union-find structure.
    parent = list(range(next_cluster_id))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    # 4. For each cell, compute its centroid and check adjacent cells (8 neighbors).
    for cell, cid in clusters.items():
        i, j = cell
        indices = cluster_points_dict[cid]
        pts = points[indices]
        centroid = pts.mean(axis=0)
        # Check neighbors: cells from (i-1,j-1) to (i+1,j+1)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                neighbor = (i + di, j + dj)
                if neighbor in clusters and neighbor != cell:
                    neighbor_cid = clusters[neighbor]
                    neighbor_pts = points[cluster_points_dict[neighbor_cid]]
                    neighbor_centroid = neighbor_pts.mean(axis=0)
                    if np.linalg.norm(centroid - neighbor_centroid) < eps:
                        union(cid, neighbor_cid)

    # 5. Merge clusters based on union-find results.
    merged = {}
    for cid in range(next_cluster_id):
        root = find(cid)
        if root not in merged:
            merged[root] = []
        merged[root].extend(cluster_points_dict[cid])

    # 6. Compute final centroids for each merged cluster.
    final_clusters = []
    for indices in merged.values():
        pts = points[indices]
        centroid = pts.mean(axis=0)
        final_clusters.append(tuple(centroid))

    return final_clusters


if __name__ == "__main__":
    # Example usage:
    # Generate synthetic data in a 1000x1000 space (for testing)
    N = 10000000  # For testing; in production, use ~10 million points.
    np.random.seed(0)
    points = np.random.rand(N, 2) * 1000  # 1000 x 1000 coordinate space
    points = points.astype(np.float64)

    eps = 5.0  # Cluster radius
    clusters = cluster_points_grid(points, eps)
    print(f"Found {len(clusters)} clusters.")
