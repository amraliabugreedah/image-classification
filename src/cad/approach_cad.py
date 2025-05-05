import ezdxf
from shapely.geometry import Polygon, Point
import os
import json


def get_entity_text(entity):
    """
    Returns the text content for either a TEXT or MTEXT entity in ezdxf.
    """
    if entity.dxftype() == "TEXT":
        # TEXT entities store their content in entity.dxf.text
        return entity.dxf.text
    elif entity.dxftype() == "MTEXT":
        # MTEXT typically has either .text or .plain_text() (depending on ezdxf version)
        # .plain_text() strips formatting codes, so it's often preferable if available.
        if hasattr(entity, "plain_text"):
            return entity.plain_text()
        else:
            # Fallback if plain_text() doesn't exist in older ezdxf versions
            return entity.text
    return ""  # default if it's neither TEXT nor MTEXT


def extract_rooms_from_dwg(dwg_path):
    doc = ezdxf.readfile(dwg_path)
    msp = doc.modelspace()

    # Query polylines and text entities
    lwpolylines = msp.query("LWPOLYLINE")
    text_entities = msp.query("TEXT MTEXT")

    rooms = []
    room_index = 0

    for lwpoly in lwpolylines:
        # Only consider closed polylines
        if lwpoly.closed:
            # Extract vertex coordinates
            coords = []
            for point in lwpoly.get_points():
                x, y = point[0], point[1]
                coords.append((x, y))

            polygon = Polygon(coords)
            area = polygon.area  # in drawing units

            inside_texts = []
            for txt in text_entities:
                # Get insertion point
                insert = txt.dxf.insert
                point = Point(insert.x, insert.y)

                if polygon.contains(point):
                    # Get the actual text content safely
                    text_value = get_entity_text(txt)
                    if text_value:
                        inside_texts.append(text_value.strip())

            # Simple heuristic: use the first text as room name
            room_name = inside_texts[0] if inside_texts else f"Room_{room_index}"

            rooms.append(
                {
                    "id": room_index,
                    "name": room_name,
                    "polygon": coords,
                    "area_raw": area,
                    "texts_found": inside_texts,
                }
            )
            room_index += 1

    return rooms


if __name__ == "__main__":
    base_dir = os.getcwd()
    dwg_file = os.path.join(base_dir, "resources", "dw", "sample.dwf")
    # output_dir = os.path.join(base_dir, "resources", "processed", "rooms", "segmented")

    extracted_rooms = extract_rooms_from_dwg(dwg_file)

    for room in extracted_rooms:
        print("Room ID:", room["id"])
        print("  Name:", room["name"])
        print("  Area (raw units):", room["area_raw"])
        print("  Texts found:", room["texts_found"])
        print("  Polygon coords:", room["polygon"])
        print("-----")
