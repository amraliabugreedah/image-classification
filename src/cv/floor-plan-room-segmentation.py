import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import measure


def segment_rooms(image_path, output_dir="segmented_rooms"):
    """
    Segment rooms from a floor plan and save each as a separate PNG file.

    Args:
        image_path (str): Path to the floor plan image
        output_dir (str): Directory to save the segmented rooms
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Create a copy for visualization
    visualization = original.copy()

    # Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to get walls and boundaries
    _, binary = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilate to close small gaps in walls
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Flood fill from outside to mark outer area
    h, w = dilated.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Use a point outside as seed (assuming the border is empty)
    cv2.floodFill(dilated, mask, (0, 0), 0)

    # Find connected components (rooms)
    labeled = measure.label(dilated, connectivity=2)
    props = measure.regionprops(labeled)

    # Filter out small regions (likely not rooms)
    min_area = 1000  # Adjust based on your floor plan scale
    room_index = 0

    # Create a colormap for visualization
    colors = plt.cm.tab20(np.linspace(0, 1, len(props)))

    for i, prop in enumerate(props):
        if prop.area >= min_area:
            # Get the room mask
            room_mask = np.zeros_like(gray)
            room_mask[labeled == prop.label] = 255

            # Get contours for the room
            contours, _ = cv2.findContours(
                room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                # Find bounding box
                x, y, w, h = cv2.boundingRect(contours[0])

                # Extract room with some padding
                padding = 10
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(original.shape[1], x + w + padding)
                y_end = min(original.shape[0], y + h + padding)

                # Create an empty image with white background
                room_img = (
                    np.ones((y_end - y_start, x_end - x_start, 3), dtype=np.uint8) * 255
                )

                # Create a mask for this specific room
                specific_mask = np.zeros_like(gray)
                cv2.drawContours(specific_mask, contours, -1, 255, -1)

                # Apply the mask to the original image
                masked_room = cv2.bitwise_and(original, original, mask=specific_mask)

                # Copy the masked room to the room image
                room_img[0 : y_end - y_start, 0 : x_end - x_start] = masked_room[
                    y_start:y_end, x_start:x_end
                ]

                # Save the room
                room_path = os.path.join(output_dir, f"room_{room_index}.png")
                cv2.imwrite(room_path, room_img)
                print(f"Saved {room_path}")

                # Draw contour on visualization with a unique color
                color = (
                    int(colors[i][2] * 255),
                    int(colors[i][1] * 255),
                    int(colors[i][0] * 255),
                )
                cv2.drawContours(visualization, contours, -1, color, 2)

                # Add room number
                cv2.putText(
                    visualization,
                    str(room_index),
                    (prop.centroid[1].astype(int), prop.centroid[0].astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                room_index += 1

    # Save the visualization
    cv2.imwrite(os.path.join(output_dir, "visualization.png"), visualization)
    print(f"Saved visualization to {os.path.join(output_dir, 'visualization.png')}")

    return room_index


def main():
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, "resources", "images", "test_6.png")
    output_dir = os.path.join(base_dir, "resources", "processed", "rooms", "segmented")


    num_rooms = segment_rooms(image_path, output_dir)
    print(f"Successfully segmented {num_rooms} rooms from the floor plan")


if __name__ == "__main__":
    main()
