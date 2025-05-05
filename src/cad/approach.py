import os
import re
import shutil
import subprocess
import tempfile
import ezdxf
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# ---------------------------------------------------------------------
# 1. Conversion (DWG/DWF -> DXF) - Placeholder Implementation
# ---------------------------------------------------------------------
def convert_to_dxf(input_file, output_file):
    """
    Attempt to convert a DWG/DWF file to DXF.
    This function first tries to run a command named 'oda_file_converter'.
    If that's not found or fails, it attempts a fallback approach.

    Adjust or remove this logic if you're already working with DXF
    or if you have another conversion solution in place.
    """
    try:
        cmd = ["oda_file_converter", input_file, os.path.dirname(output_file)]
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print(
            "Converter tool 'oda_file_converter' not found. Checking for fallback DXF..."
        )
    except subprocess.CalledProcessError as e:
        print(f"Conversion command failed: {e}")

    # Fallback: for testing, see if there's a pre-converted DXF in 'sample_dxfs' folder
    fallback_dir = os.path.join(os.path.dirname(__file__), "sample_dxfs")
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    fallback_dxf = os.path.join(fallback_dir, f"{base_name}.dxf")
    if os.path.exists(fallback_dxf):
        shutil.copy(fallback_dxf, output_file)
        print(f"Using fallback DXF: {fallback_dxf}")
        return True
    else:
        print(f"No fallback DXF found for {input_file}.")
        return False


# ---------------------------------------------------------------------
# 2. Geometric Utility: Point-in-Polygon (Ray Casting)
# ---------------------------------------------------------------------
def point_in_polygon(point, polygon):
    """
    Determine if a 2D point is inside a polygon using the ray casting algorithm.
    """
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


# ---------------------------------------------------------------------
# 3. Parsing & Extraction
# ---------------------------------------------------------------------
class CADParser:
    """
    Converts DWG/DWF -> DXF (if needed) and extracts room polygons.

    By default, we only look for polylines on layers like "A-ROOM" or "FLOOR_PLAN".
    We skip the "ROOM INFORMATION" layer or small polygons (area < threshold)
    to avoid annotation bounding boxes.
    """

    def __init__(self, file_path):
        self.original_file = file_path
        self.dxf_file = file_path
        self.doc = None
        self.msp = None
        self.rooms = []
        self.doors = []
        self.windows = []
        self.texts = []

        # Adjust these to match your CAD file's real room boundary layers:
        self.allowed_room_layers = {"A-ROOM", "FLOOR_PLAN"}
        # If you want to skip all "ROOM INFORMATION" polylines:
        self.skip_layers = {"ROOM INFORMATION"}

        # If you want to skip polygons with area < some threshold (e.g., 2.0)
        self.min_area_threshold = 2.0

    def prepare_file(self):
        """
        If input is DWG/DWF, convert it to DXF. Otherwise assume it's already DXF.
        """
        _, ext = os.path.splitext(self.original_file)
        ext = ext.lower()
        if ext in [".dwg", ".dwf"]:
            temp_dir = tempfile.gettempdir()
            base_name = os.path.splitext(os.path.basename(self.original_file))[0]
            self.dxf_file = os.path.join(temp_dir, f"{base_name}.dxf")
            print(f"Converting {self.original_file} to DXF: {self.dxf_file}")
            success = convert_to_dxf(self.original_file, self.dxf_file)
            if not success:
                raise Exception("File conversion failed.")
        elif ext == ".dxf":
            self.dxf_file = self.original_file
        else:
            raise Exception("Unsupported file format. Provide a DWG, DWF, or DXF file.")

    def load_document(self):
        """
        Load the DXF file and get the modelspace.
        """
        try:
            self.doc = ezdxf.readfile(self.dxf_file)
            self.msp = self.doc.modelspace()
        except IOError as e:
            raise IOError(f"Error opening file '{self.dxf_file}': {e}")
        except ezdxf.DXFStructureError as e:
            raise ValueError(f"Invalid DXF file '{self.dxf_file}': {e}")

    def parse_entities(self):
        """
        Extract polylines that represent real rooms, ignoring annotation bounding boxes.
        Also demonstrate how you might extract doors/windows or text if needed.
        """
        if self.msp is None:
            return

        for entity in self.msp:
            etype = entity.dxftype()

            # --- Polylines (Potential Rooms) ---
            if etype in ("LWPOLYLINE", "POLYLINE") and entity.closed:
                layer_name = entity.dxf.layer.upper()

                # 1) Skip known annotation layers
                if layer_name in self.skip_layers:
                    continue

               

                # Extract points
                points = (
                    list(entity.get_points("xy"))
                    if etype == "LWPOLYLINE"
                    else [(pt[0], pt[1]) for pt in entity.get_points()]
                )

                # Compute area
                try:
                    area = entity.get_area()
                except Exception:
                    area = None

                # Skip small polygons
                if area is not None and area < self.min_area_threshold:
                    continue

                # Add to rooms
                self.rooms.append(
                    {
                        "layer": entity.dxf.layer,
                        "area": area,
                        "points": points,
                        "name": None,
                    }
                )

            # --- INSERT (Potential Doors/Windows) ---
            elif etype == "INSERT":
                block_name = entity.dxf.name.upper()
                if "DOOR" in block_name:
                    self.doors.append(
                        {
                            "block": block_name,
                            "insertion_point": entity.dxf.insert,
                        }
                    )
                elif "WIN" in block_name or "WINDOW" in block_name:
                    self.windows.append(
                        {
                            "block": block_name,
                            "insertion_point": entity.dxf.insert,
                        }
                    )

            # --- TEXT / MTEXT ---
            elif etype in ["TEXT", "MTEXT"]:
                text_value = (
                    entity.plain_text() if etype == "MTEXT" else entity.dxf.text
                )
                self.texts.append(
                    {
                        "text": text_value,
                        "position": entity.dxf.insert,
                    }
                )

    def parse(self):
        """
        Full parse routine.
        1) Convert if needed
        2) Load doc
        3) Parse relevant entities
        4) Return feature dictionary
        """
        self.prepare_file()
        self.load_document()
        self.parse_entities()
        return {
            "rooms": self.rooms,
            "doors": self.doors,
            "windows": self.windows,
            "texts": self.texts,
        }


# ---------------------------------------------------------------------
# 4. Feature Interpretation
# ---------------------------------------------------------------------
class FeatureInterpreter:
    """
    Assign room names by matching text inside the polygons (optional).
    Compute centroid for grouping or further logic.
    """

    def __init__(self, features):
        self.features = features

    def assign_room_names(self):
        rooms = self.features.get("rooms", [])
        texts = self.features.get("texts", [])

        for room in rooms:
            poly = room["points"]
            assigned_texts = []

            # For each text entity, check if it falls inside the polygon
            for txt in texts:
                pos = txt["position"]
                if point_in_polygon((pos[0], pos[1]), poly):
                    assigned_texts.append(txt["text"])

            # If any text found inside, name the room by concatenating them
            if assigned_texts:
                room["name"] = " / ".join(assigned_texts)
            else:
                # Fallback name: "Room_<centroid>"
                cx = sum(x for x, y in poly) / len(poly)
                cy = sum(y for x, y in poly) / len(poly)
                room["name"] = f"Room_{cx:.1f}_{cy:.1f}"

            # Store centroid for grouping
            room["centroid"] = (
                sum(x for x, y in poly) / len(poly),
                sum(y for x, y in poly) / len(poly),
            )

        self.features["rooms"] = rooms
        return self.features


# ---------------------------------------------------------------------
# 5. Grouping Logic (By Floor, if needed)
# ---------------------------------------------------------------------
def group_rooms_by_floor(rooms, threshold=5.0):
    """
    Example grouping: We cluster rooms by the Y-coordinate of their centroid.
    If two rooms have centroids within 'threshold' units on Y, they go in the same group.

    This is just a heuristic. You might have an explicit floor attribute or a different approach.
    """
    sorted_rooms = sorted(rooms, key=lambda r: r["centroid"][1])
    floor_groups = {}
    current_group = []
    current_floor_value = None

    for room in sorted_rooms:
        cy = room["centroid"][1]
        if current_floor_value is None:
            current_floor_value = cy
            current_group.append(room)
        else:
            if abs(cy - current_floor_value) <= threshold:
                current_group.append(room)
            else:
                floor_label = f"Floor_{int(current_floor_value)}"
                floor_groups[floor_label] = current_group
                current_group = [room]
                current_floor_value = cy

    # Add the last group
    if current_group:
        floor_label = f"Floor_{int(current_floor_value)}"
        floor_groups[floor_label] = current_group

    return floor_groups


# ---------------------------------------------------------------------
# 6. Visualization & PDF Generation
# ---------------------------------------------------------------------
def create_room_image(room, output_dir=None):
    """
    Draw a single room polygon with matplotlib and save as PNG.
    Returns the path to the saved image file.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    points = room.get("points", [])

    if points:
        poly_patch = Polygon(
            points,
            closed=True,
            fill=True,
            edgecolor="black",
            facecolor="lightgreen",
            alpha=0.5,
        )
        ax.add_patch(poly_patch)

        # Annotate with the room name
        cx, cy = room["centroid"]
        ax.text(
            cx,
            cy,
            room.get("name", ""),
            fontsize=10,
            ha="center",
            va="center",
            color="black",
        )

        xs, ys = zip(*points)
        ax.set_xlim(min(xs) - 1, max(xs) + 1)
        ax.set_ylim(min(ys) - 1, max(ys) + 1)

    ax.set_aspect("equal")
    ax.axis("off")

    if output_dir is None:
        output_dir = tempfile.gettempdir()
    os.makedirs(output_dir, exist_ok=True)

    # Clean up the room name for filename
    safe_name = (
        room.get("name", "room").replace("/", "_").replace(" ", "_").replace("\n", "_")
    )
    image_path = os.path.join(output_dir, f"{safe_name}.png")

    plt.savefig(image_path, bbox_inches="tight")
    plt.close(fig)
    return image_path


def create_pdf_for_floor(floor_label, rooms, output_dir="resources/features"):
    """
    Creates a PDF for a specific floor. Each page = one room image.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{floor_label}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    page_width, page_height = letter

    for room in rooms:
        image_path = create_room_image(room)
        # Place the image to fill the page
        c.drawImage(image_path, 0, 0, width=page_width, height=page_height)
        c.showPage()
    c.save()

    print(f"PDF for {floor_label} saved to {pdf_path}")


# ---------------------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = os.getcwd()

    # Path to the input floor plan image
    # Replace with the path to your DWG/DWF file (or DXF file)
    input_file_path = os.path.join(base_dir, "resources", "dw", "sample.dxf")

    # 1. Parse & Extract
    parser = CADParser(input_file_path)
    try:
        extracted_features = parser.parse()
    except Exception as e:
        print(f"Error during parsing: {e}")
        exit(1)

    # 2. Interpret (Name rooms, compute centroids, etc.)
    interpreter = FeatureInterpreter(extracted_features)
    interpreted_features = interpreter.assign_room_names()

    # 3. (Optional) Group by floor
    rooms = interpreted_features.get("rooms", [])
    floor_groups = group_rooms_by_floor(rooms, threshold=5.0)
    print(rooms)
    # 4. Create a PDF per floor
    for floor_label, room_list in floor_groups.items():
        create_pdf_for_floor(floor_label, room_list)

    # Print the interpreted features for demonstration
    # print("Extracted and Interpreted CAD Features:")
    # print("Rooms:")
    # for room in interpreted_features["rooms"]:
    #     print(f"  Name: {room['name']}, Area: {room['area']}, Points: {room['points']}")
    # print("Doors:")
    # for door in interpreted_features["doors"]:
    #     print(f"  Block: {door['block']}, Insertion: {door['insertion_point']}")
    # print("Windows:")
    # for window in interpreted_features["windows"]:
    #     print(f"  Block: {window['block']}, Insertion: {window['insertion_point']}")
    # print("Texts:")
    # for text in interpreted_features["texts"]:
    #     print(f"  Text: {text['text']}, Position: {text['position']}")
