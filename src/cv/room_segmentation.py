import os
import cv2
import numpy as np
import math
import logging
import pytesseract
from pytesseract import Output
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def mask_text_regions(gray_image, conf_threshold=60):
    """
    Use Tesseract to detect text bounding boxes in 'gray_image' (grayscale).
    Fill those bounding boxes with black if confidence > conf_threshold.
    """
    logger.debug("Starting text masking with conf_threshold=%d", conf_threshold)
    data = pytesseract.image_to_data(
        gray_image, config="--psm 11 --oem 3", output_type=Output.DICT
    )
    masked = gray_image.copy()
    n_boxes = len(data["level"])
    for i in range(n_boxes):
        try:
            conf = int(data["conf"][i])
        except:
            # Tesseract can produce non-integer conf
            continue
        if conf > conf_threshold:
            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )
            cv2.rectangle(masked, (x, y), (x + w, y + h), 0, thickness=-1)
    logger.debug("Text masking complete.")
    return masked


def detect_and_segment_rooms(
    image_path, output_dir, min_area=5000, merge_dist=10, angle_thresh=5
):
    """
    Full pipeline to:
      1) Mask text
      2) Morphologically close the image
      3) Detect lines via Hough
      4) Merge lines
      5) Find intersections & build graph
      6) Detect cycles as rooms
      7) Filter by area & segment each room

    Args:
        image_path (str): Path to the input floor plan image.
        output_dir (str): Directory to save segmented room images.
        min_area (float): Minimum polygon area for a valid room.
        merge_dist (float): Distance threshold for merging lines.
        angle_thresh (float): Angle threshold for merging lines.
    Returns:
        list of dict: Each entry has:
          - "polygon": list of (x, y) vertices
          - "area": area of the polygon
          - "bounding_box": (x, y, w, h)
    """
    logger.debug("Loading image from %s", image_path)
    original = cv2.imread(image_path)
    if original is None:
        logger.error("Could not load image at %s", image_path)
        raise FileNotFoundError(f"Could not load image at {image_path}")
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    logger.debug("Image converted to grayscale.")

    # 1) Mask text
    gray_masked = mask_text_regions(gray, conf_threshold=60)

    # 2) Threshold & morphological closing
    _, thresh = cv2.threshold(
        gray_masked, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    logger.debug("Thresholding complete. Thresh shape: %s", thresh.shape)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    logger.debug("Morphological closing complete.")

    # 3) Edge detection
    edges = cv2.Canny(closed, 50, 150, apertureSize=3)
    logger.debug("Canny edge detection complete.")

    # 4) Hough lines
    lines_p = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10
    )
    if lines_p is None:
        logger.warning("No lines detected.")
        return []
    lines = []
    for ln in lines_p:
        x1, y1, x2, y2 = ln[0]
        lines.append(((x1, y1), (x2, y2)))
    logger.debug("Detected %d raw lines from Hough.", len(lines))

    # 5) Merge lines
    merged_lines = merge_lines(lines, dist_thresh=merge_dist, angle_thresh=angle_thresh)
    logger.debug("Merged lines count: %d", len(merged_lines))

    # 6) Find intersections & cluster them
    intersections = []
    for i in range(len(merged_lines)):
        for j in range(i + 1, len(merged_lines)):
            pt = line_intersection(merged_lines[i], merged_lines[j])
            if pt is not None:
                intersections.append(pt)
    logger.debug("Found %d intersections (unclustered).", len(intersections))
    intersection_array = np.array(intersections, dtype=np.float64)
    unique_pts = cluster_points_dbscan(intersection_array, eps=5.0, min_samples=1)
    logger.debug("Unique intersection points after clustering: %d", len(unique_pts))

    # 7) Build graph & find cycles
    graph = build_graph_parallel(
        merged_lines, unique_pts, tol=5, n_workers=os.cpu_count()
    )
    logger.debug("Graph built with %d nodes.", len(unique_pts))
    polygons = find_polygons_in_graph(graph, unique_pts, max_length=10)
    logger.debug("Detected %d polygons before area filtering.", len(polygons))

    # 8) Filter polygons by area & segment
    results = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, poly in enumerate(polygons):
        area = polygon_area(poly)
        if area < min_area:
            continue
        minx = min(p[0] for p in poly)
        maxx = max(p[0] for p in poly)
        miny = min(p[1] for p in poly)
        maxy = max(p[1] for p in poly)
        box = (int(minx), int(miny), int(maxx - minx), int(maxy - miny))

        # Crop the original image to get the segmented "room"
        x, y, w, h = box
        room_crop = original[y : y + h, x : x + w]

        # Save the segmented room
        output_path = os.path.join(output_dir, f"room_{i+1}.png")
        cv2.imwrite(output_path, room_crop)
        logger.debug(
            "Saved room %d => %s; area=%.2f, box=%s", i + 1, output_path, area, box
        )

        results.append({"polygon": poly, "area": area, "bounding_box": box})

    logger.info(
        "Segmentation complete. Found %d room polygons above area threshold.",
        len(results),
    )
    return results


# ------------------------
# HELPER FUNCTIONS
# ------------------------
def merge_lines(lines, dist_thresh=10, angle_thresh=5):
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
        used[i] = True
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            l2 = lines[j]
            angle2, length2 = line_to_polar(l2)
            if abs(angle_diff(angle1, angle2)) < angle_thresh:
                if lines_are_close(l1, l2, dist_thresh):
                    used[j] = True
        merged.append(l1)
    return merged


def line_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    x3, y3, x4, y4 = map(float, (x3, y3, x4, y4))
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-7:
        return None
    ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return (ix, iy)


def angle_diff(a1, a2):
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff


def lines_are_close(l1, l2, dist_thresh=10):
    (x1, y1), (x2, y2) = l1
    (x3, y3), (x4, y4) = l2
    mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
    mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
    return math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1]) < dist_thresh


# -----------
# DBSCAN CLUSTERING
# -----------
def cluster_points_dbscan(points, eps=5.0, min_samples=1):
    if len(points) == 0:
        return np.empty((0, 2))
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_pts = points[labels == label]
        centroid = cluster_pts.mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids, dtype=np.float64)


# -----------
# BUILD GRAPH (Parallel) & FIND POLYGONS
# -----------
def vectorized_distance_to_segment(pts, line):
    (x1, y1), (x2, y2) = line
    pts = np.asarray(pts, dtype=np.float64)
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return np.linalg.norm(pts - np.array([x1, y1]), axis=1)
    t = ((pts[:, 0] - x1) * dx + (pts[:, 1] - y1) * dy) / denom
    t = np.clip(t, 0, 1)
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.sqrt((pts[:, 0] - proj_x) ** 2 + (pts[:, 1] - proj_y) ** 2)


def distance_along_line(pt, line):
    (x1, y1), (x2, y2) = line
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom == 0:
        return 0
    return ((pt[0] - x1) * dx + (pt[1] - y1) * dy) / denom


def process_line_for_graph(line, unique_pts, tree, tol):
    (x1, y1), (x2, y2) = line
    x_min = min(x1, x2) - tol
    x_max = max(x1, x2) + tol
    y_min = min(y1, y2) - tol
    y_max = max(y1, y2) + tol
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    radius = math.hypot(x_max - x_min, y_max - y_min) / 2
    candidate_indices = tree.query_ball_point(center, r=radius)
    # Filter by bounding box
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
    candidate_pts = unique_pts[candidates]
    dists = vectorized_distance_to_segment(candidate_pts, line)
    nodes_on_line = [candidates[i] for i, d in enumerate(dists) if d < tol]
    if len(nodes_on_line) < 2:
        return []
    nodes_on_line.sort(key=lambda idx: distance_along_line(unique_pts[idx], line))
    edges = []
    for a, b in zip(nodes_on_line, nodes_on_line[1:]):
        edges.append((a, b))
    return edges


def build_graph_parallel(merged_lines, unique_pts, tol=5, n_workers=4):
    from scipy.spatial import cKDTree

    tree = cKDTree(unique_pts)
    all_edges = []
    func = partial(process_line_for_graph, unique_pts=unique_pts, tree=tree, tol=tol)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(func, line) for line in merged_lines]
        for f in as_completed(futures):
            edges = f.result()
            if edges:
                all_edges.extend(edges)
    logger.debug("Total edges collected: %d", len(all_edges))
    graph = {i: [] for i in range(len(unique_pts))}
    for a, b in all_edges:
        if b not in graph[a]:
            graph[a].append(b)
        if a not in graph[b]:
            graph[b].append(a)
    return graph


def find_polygons_in_graph(graph, points, max_length=10):
    polygons = []

    def dfs_cycle(path, current, start):
        if len(path) > max_length:
            return
        for nxt in graph[current]:
            if nxt == start and len(path) > 2:
                polygons.append(path)
            elif nxt not in path:
                dfs_cycle(path + [nxt], nxt, start)

    for node in graph.keys():
        dfs_cycle([node], node, node)
    # Filter duplicates
    unique_polys = []
    seen = set()
    for cycle in polygons:
        sorted_cycle = tuple(sorted(cycle))
        if sorted_cycle not in seen:
            seen.add(sorted_cycle)
            poly_points = [points[n] for n in cycle]
            poly_points = order_polygon(poly_points)
            unique_polys.append(poly_points)
    return unique_polys


def order_polygon(points):
    if not points:
        return points
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    def angle_from_center(p):
        return math.atan2(p[1] - cy, p[0] - cx)

    return sorted(points, key=angle_from_center)


def polygon_area(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - y1 * x2
    return abs(area) / 2.0


# ------------------------
# EXAMPLE MAIN
# ------------------------
if __name__ == "__main__":
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, "resources", "images", "test_6.png")
    output_dir = os.path.join(base_dir, "resources", "processed", "rooms", "segmented")
    os.makedirs(output_dir, exist_ok=True)

    results = detect_and_segment_rooms(
        image_path=image_path,
        output_dir=output_dir,
        min_area=1,
        merge_dist=10,
        angle_thresh=5,
    )
    logger.info("Final: %d segmented rooms saved.", len(results))
