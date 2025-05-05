import base64
import os
import json


def save_base64_images(image_list, output_dir="resources/images/base64toimg"):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for index, image_str in enumerate(image_list):
        # Remove the data URI scheme if present (e.g., "data:image/png;base64,")
        if image_str.startswith("data:"):
            header, image_str = image_str.split(",", 1)

        try:
            # Decode the base64 string
            image_bytes = base64.b64decode(image_str)
        except Exception as e:
            print(f"Failed to decode image at index {index}: {e}")
            continue

        # Create a file path for each image
        output_path = os.path.join(output_dir, f"img_{index}.png")
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        print(f"Saved image {index} to: {output_path}")


# Example usage:

base_dir = os.getcwd()
rooms_output_dir = os.path.join(base_dir, "resources", "images", "base64toimg")

# Read data from the specified JSON file
with open(
    "/Users/greedah/Documents/ai71/code/cons71/playground/req-from-models.json", "r"
) as file:
    data = json.load(file)

save_base64_images(data["images"], rooms_output_dir)
