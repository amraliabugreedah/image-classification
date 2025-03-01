import os
import cv2
import numpy as np
import math
import itertools
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from src.utils.grid_clustering import cluster_points_grid

max_workers = os.cpu_count()
print(max_workers)


def detect_rooms_hough(image_path, min_area=5000, merge_dist=10, angle_thresh=5):
    """
    Detect rooms in a floor plan using a Hough Transform–based approach.
    1) Canny edge detection
    2) Probabilistic Hough Transform to detect line segments at any angle
    3) Merge lines that are nearly collinear and close
    4) Find intersections to form graph nodes
    5) Attempt to build polygons (cycles) from the graph
    6) Filter polygons by area and return bounding boxes (or other info)

    Args:
        image_path (str): Path to the input floor plan image.
        min_area (float): Minimum polygon area to be considered a valid room.
        merge_dist (float): Max distance (in pixels) to consider lines for merging.
        angle_thresh (float): Max angle difference (in degrees) for merging lines.

    Returns:
        list of dict: Each dict may contain:
          - "polygon": list of (x, y) vertices
          - "area": float area of the polygon
          - "bounding_box": (x, y, w, h)
    """
    # 1) Load image and detect edges
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Optional: upsample if lines are too thin
    # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Canny edge detection
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
    print("step", 2)

    # 2) Probabilistic Hough Transform
    lines_p = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
    )
    if lines_p is None:
        print("No lines detected.")
        return []

    # Convert lines to a more convenient structure
    # lines_p is shape [N,1,4] => x1,y1,x2,y2
    lines = []
    for ln in lines_p:
        x1, y1, x2, y2 = ln[0]
        lines.append(((x1, y1), (x2, y2)))
    print("step", 3)

    # 3) Merge lines that are nearly collinear and close
    merged_lines = merge_lines(lines, dist_thresh=merge_dist, angle_thresh=angle_thresh)
    print("step", 4)

    # 4) Find intersections => graph nodes
    intersections = []
    for i in range(len(merged_lines)):
        for j in range(i + 1, len(merged_lines)):
            pt = line_intersection(merged_lines[i], merged_lines[j])
            if pt is not None:
                intersections.append(pt)

    print("step", 4, "*")
    intersection_array = np.array(intersections, dtype=np.float64)

    # Cluster the intersections using the grid-based method with your chosen eps
    unique_pts = cluster_points_grid(intersection_array, eps=5.0)

    print("step", 5)

    # 5) Build a graph: each intersection is a node
    #    For each line, connect nodes that lie on it
    graph = build_graph_parallel(merged_lines, unique_pts, tol=5)

    print("step", 6)

    # 6) Find cycles in the graph => polygons
    #    We'll do a simple cycle detection. This is a big topic; we do a naive approach.
    polygons = find_polygons_in_graph(graph, unique_pts)
    print("step", 7)
    # 7) Filter polygons by area
    results = []
    for poly in polygons:
        area = polygon_area(poly)
        if area < min_area:
            continue
        # bounding box
        minx = min(p[0] for p in poly)
        maxx = max(p[0] for p in poly)
        miny = min(p[1] for p in poly)
        maxy = max(p[1] for p in poly)
        results.append(
            {
                "polygon": poly,
                "area": area,
                "bounding_box": (
                    int(minx),
                    int(miny),
                    int(maxx - minx),
                    int(maxy - miny),
                ),
            }
        )

    return results


def merge_lines(lines, dist_thresh=10, angle_thresh=5):
    """
    Merge lines that are nearly collinear (angle < angle_thresh)
    and whose endpoints are within dist_thresh of each other.
    This is a very rough approach; better merging requires more advanced logic.
    """
    merged = []
    used = [False] * len(lines)

    def line_to_polar(line):
        (x1, y1), (x2, y2) = line
        dx, dy = (x2 - x1), (y2 - y1)
        angle = math.degrees(math.atan2(dy, dx))
        length = math.hypot(dx, dy)
        return angle, length

    for i in range(len(lines)):
        if used[i]:
            continue
        l1 = lines[i]
        angle1, length1 = line_to_polar(l1)
        group = [l1]
        used[i] = True

        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            l2 = lines[j]
            angle2, length2 = line_to_polar(l2)

            # Compare angles
            if abs(angle_diff(angle1, angle2)) < angle_thresh:
                # Check if lines are close
                if lines_are_close(l1, l2, dist_thresh):
                    group.append(l2)
                    used[j] = True

        # TODO: Actually merge the lines in 'group'
        # For now, we just pick one as representative
        merged.append(l1)

    return merged


def line_intersection(line1, line2):
    """
    Return the intersection point (x, y) of two infinite lines,
    or None if they are parallel or nearly parallel.
    Each line is ((x1, y1), (x2, y2)).
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    x3, y3, x4, y4 = map(float, (x3, y3, x4, y4))

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-7:
        return None  # Parallel or nearly parallel

    intersect_x = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    intersect_y = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom

    return (intersect_x, intersect_y)


def point_on_line(pt, line, tol=5):
    """
    Check if point 'pt' lies close to line segment 'line' within tolerance 'tol'.
    """
    dist = distance_to_segment(pt, line)
    return dist < tol


def distance_to_segment(pt, line):
    """
    Return min distance from point 'pt' to the line segment 'line'.
    """
    (x1, y1), (x2, y2) = line
    px, py = pt

    # Parametric t of projection
    dx, dy = (x2 - x1), (y2 - y1)
    if dx == 0 and dy == 0:
        return distance(pt, (x1, y1))
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    if t < 0:
        return distance(pt, (x1, y1))
    elif t > 1:
        return distance(pt, (x2, y2))
    projx = x1 + t * dx
    projy = y1 + t * dy
    return distance(pt, (projx, projy))


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def lines_are_close(l1, l2, dist_thresh=10):
    """
    Check if two lines are within dist_thresh of each other
    by checking endpoint distances or some other heuristic.
    """
    # Quick approach: check distance between midpoints
    mid1 = ((l1[0][0] + l1[1][0]) / 2, (l1[0][1] + l1[1][1]) / 2)
    mid2 = ((l2[0][0] + l2[1][0]) / 2, (l2[0][1] + l2[1][1]) / 2)
    if distance(mid1, mid2) < dist_thresh:
        return True
    return False


def angle_diff(a1, a2):
    """
    Return the minimal difference between angles a1, a2 in degrees.
    """
    diff = abs(a1 - a2) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def find_polygons_in_graph(graph, points):
    """
    Find cycles (polygons) in the adjacency list 'graph'.
    Each node in 'graph' is an index into 'points'.
    Return a list of polygons, each polygon is a list of (x, y).

    This is a naive approach that might find duplicates or partial cycles.
    """
    visited = set()
    polygons = []

    def dfs_cycle(path, current, start):
        # Attempt to find a cycle that returns to 'start'
        for nxt in graph[current]:
            if nxt == start and len(path) > 2:
                # Found a cycle
                cycle = path[:]  # copy
                polygons.append(cycle)
            elif nxt not in path:
                # Continue DFS
                dfs_cycle(path + [nxt], nxt, start)

    # Try each node as a potential cycle start
    for node in graph.keys():
        dfs_cycle([node], node, node)

    # Convert node indices to coordinates
    # We might have duplicates or reversed cycles
    # We'll do a quick uniqueness pass
    unique_polys = []
    seen = set()

    for cycle in polygons:
        # Sort cycle node indices
        sorted_cycle = sorted(cycle)
        t = tuple(sorted_cycle)
        if t not in seen:
            seen.add(t)
            # Convert to points
            poly_points = [points[n] for n in cycle]
            # We can reorder them in a consistent winding order if we want
            poly_points = order_polygon(poly_points)
            unique_polys.append(poly_points)

    return unique_polys


def order_polygon(points):
    """
    Order polygon vertices in a clockwise or counter-clockwise manner.
    A naive approach: compute centroid, then sort by angle from centroid.
    """
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    def angle_from_center(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    pts_sorted = sorted(points, key=angle_from_center)
    return pts_sorted


def polygon_area(poly):
    """
    Compute polygon area using the Shoelace formula.
    """
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0


# Vectorized distance function: computes distance from each point in 'pts' to the line segment.
def vectorized_distance_to_segment(pts, line):
    """
    Compute the distance from each point in pts to the given line segment.

    Args:
        pts (np.ndarray): Array of shape (N, 2) containing point coordinates.
        line (tuple): ((x1, y1), (x2, y2)) defining the line segment.

    Returns:
        np.ndarray: Distances for each point.
    """
    (x1, y1), (x2, y2) = line
    # Ensure we work in float64 for precision
    pts = np.asarray(pts, dtype=np.float64)
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    dx = x2 - x1
    dy = y2 - y1
    # Squared length of the segment
    denom = dx * dx + dy * dy
    if denom == 0:
        # If the line segment is a point
        return np.linalg.norm(pts - np.array([x1, y1]), axis=1)
    # Compute the projection parameter t for all points
    t = ((pts[:, 0] - x1) * dx + (pts[:, 1] - y1) * dy) / denom
    t = np.clip(t, 0, 1)  # Restrict t to the segment [0,1]
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    distances = np.sqrt((pts[:, 0] - proj_x) ** 2 + (pts[:, 1] - proj_y) ** 2)
    return distances


def distance_along_line(pt, line):
    """Return the projection parameter t (distance along line normalized) for point pt on line."""
    (x1, y1), (x2, y2) = line
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return 0
    t = ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / (dx * dx + dy * dy)
    return t


def process_line_for_graph(line, unique_pts, tree, tol):
    """
    Process one merged line:
      - Use cKDTree to get candidate unique points that are near the line.
      - Compute distances for these candidates.
      - Return a list of edges (pairs of node indices) for points that lie on the line.
    """
    # Unpack line endpoints
    (x1, y1), (x2, y2) = line

    # Define an expanded bounding box around the line (with tolerance)
    x_min = min(x1, x2) - tol
    x_max = max(x1, x2) + tol
    y_min = min(y1, y2) - tol
    y_max = max(y1, y2) + tol
    center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
    radius = math.hypot(x_max - x_min, y_max - y_min) / 2.0

    # Query the tree for candidate indices within the circle
    candidate_indices = tree.query_ball_point(center, r=radius)
    # Filter candidates by the rectangle boundaries (faster than looping over 10M points)
    candidates = [
        i
        for i in candidate_indices
        if (
            unique_pts[i, 0] >= x_min
            and unique_pts[i, 0] <= x_max
            and unique_pts[i, 1] >= y_min
            and unique_pts[i, 1] <= y_max
        )
    ]
    if len(candidates) == 0:
        return []
    candidate_points = unique_pts[candidates]

    # Compute distances for candidate points from the line
    dists = vectorized_distance_to_segment(candidate_points, line)
    # Select points with distance less than tolerance
    nodes_on_line = [candidates[i] for i, d in enumerate(dists) if d < tol]
    if len(nodes_on_line) < 2:
        return []

    # Sort nodes_on_line by their distance along the line
    nodes_on_line.sort(key=lambda idx: distance_along_line(unique_pts[idx], line))

    # Build edges between consecutive nodes
    edges = []
    for a, b in zip(nodes_on_line, nodes_on_line[1:]):
        edges.append((a, b))
    return edges


def build_graph_parallel(merged_lines, unique_pts, tol=5, n_workers=12):
    """
    Build a graph (adjacency list) of intersection nodes along merged lines,
    using spatial indexing and parallel processing.

    Args:
        merged_lines (list): List of merged line segments (each defined by ((x1, y1), (x2, y2))).
        unique_pts (np.ndarray): Array of shape (N,2) of unique intersection points (float64).
        tol (float): Tolerance for determining if a point is on a line.
        n_workers (int): Number of parallel workers.

    Returns:
        dict: Graph represented as a dictionary mapping node index to a list of connected node indices.
    """
    # Build a spatial index using cKDTree
    tree = cKDTree(unique_pts)

    all_edges = []
    # Use ProcessPoolExecutor to process each line in parallel.
    func = partial(process_line_for_graph, unique_pts=unique_pts, tree=tree, tol=tol)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(func, line) for line in merged_lines]
        for f in as_completed(futures):
            edges = f.result()
            if edges:
                all_edges.extend(edges)
    print("step 5 here 1")
    # Build graph from all_edges
    graph = {i: [] for i in range(len(unique_pts))}
    for a, b in all_edges:
        if b not in graph[a]:
            graph[a].append(b)
        if a not in graph[b]:
            graph[b].append(a)
    return graph


if __name__ == "__main__":
    base_dir = os.getcwd()
    rooms_output_dir = os.path.join(
        base_dir, "resources", "processed", "rooms", "hough"
    )

    # Path to the input floor plan image
    image_path = os.path.join(base_dir, "resources", "images", "sample_plan.png")

    os.makedirs(rooms_output_dir, exist_ok=True)
    # Example usage
    test_image_path = os.path.join(os.getcwd(), "resources", "images", "page_2.png")
    rooms = detect_rooms_hough(test_image_path, min_area=5000)

    original = cv2.imread(image_path)

    for i, room in enumerate(rooms):
        # Get bounding box from detected room polygon
        x, y, w, h = room["bounding_box"]

        # Crop the original image using the bounding box
        room_crop = original[y : y + h, x : x + w]

        # Save the cropped room image to the rooms folder
        output_path = os.path.join(rooms_output_dir, f"room_{i+1}.png")
        cv2.imwrite(output_path, room_crop)
        print(f"Room {i+1} saved to {output_path}")

    print(f"Detected {len(rooms)} polygons above area threshold.")

    for i, r in enumerate(rooms):
        print(f"Polygon {i+1}: area={r['area']:.2f}, bounding_box={r['bounding_box']}")
