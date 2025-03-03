import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def segment_rooms_from_simple_floorplan(image_path, output_dir="room_images"):
    """
    Segment rooms from a simple line-drawing floor plan image,
    and save each room as an individual image file.
    
    Parameters:
    image_path (str): Path to the floor plan image
    output_dir (str): Directory to save individual room images
    
    Returns:
    tuple: (original image, labeled rooms image, dictionary of rooms with their labels)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    print(f"Image dimensions: {img.shape}")
    
    # Create a copy for visualization
    original = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Debug save
    cv2.imwrite("1_gray.png", gray)
    
    # Threshold the image - for line drawings we need a different approach
    # First, invert the image so walls are white
    _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Debug save
    cv2.imwrite("2_binary.png", binary)
    
    # Dilate the walls slightly to ensure they're connected
    kernel = np.ones((3, 3), np.uint8)
    dilated_walls = cv2.dilate(binary, kernel, iterations=1)
    
    # Debug save
    cv2.imwrite("3_dilated_walls.png", dilated_walls)
    
    # Create a larger image with a border to ensure outer rooms are detected
    h, w = dilated_walls.shape
    bordered = np.zeros((h+40, w+40), dtype=np.uint8)
    bordered[20:h+20, 20:w+20] = dilated_walls
    
    # Add a border around the entire image
    cv2.rectangle(bordered, (10, 10), (w+30, h+30), 255, 2)
    
    # Debug save
    cv2.imwrite("4_bordered.png", bordered)
    
    # Detect rooms using connected components on the inverse
    # (the spaces between walls are rooms)
    room_mask = cv2.bitwise_not(bordered)
    
    # Remove small holes inside rooms
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Debug save
    cv2.imwrite("5_room_mask.png", room_mask)
    
    # Label connected components (rooms)
    labeled_array, num_features = ndimage.label(room_mask)
    print(f"Found {num_features} potential rooms")
    
    # Create colored image for visualization
    colored_rooms = np.zeros((h+40, w+40, 3), dtype=np.uint8)
    
    # Random colors for rooms
    colors = np.random.randint(0, 255, size=(num_features + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    
    for i in range(1, num_features + 1):
        colored_rooms[labeled_array == i] = colors[i]
    
    # Debug save
    cv2.imwrite("6_colored_rooms.png", colored_rooms)
    
    # Trim colored_rooms back to original size
    colored_rooms_trimmed = colored_rooms[20:h+20, 20:w+20]
    cv2.imwrite("7_colored_rooms_trimmed.png", colored_rooms_trimmed)
    
    # Define room names manually for this specific floor plan
    # This is better than OCR for this type of drawing
    # You'll need to adjust these coordinates based on your specific floor plan
    room_labels = {
        "MAID ROOM": (100, 80),  # Approximate (x, y) center of the room
        "IRON": (420, 80),
        "LAUNDRY": (420, 180),
        "BATHROOM": (200, 200)
    }
    
    # Create a result dictionary with room information
    rooms_info = {}
    
    # Process each room
    for room_id in range(1, num_features + 1):
        # Create a mask for the current room in the original image coordinates
        room_mask_original = (labeled_array[20:h+20, 20:w+20] == room_id)
        
        # Skip if the room is too small (likely noise)
        room_area = np.sum(room_mask_original)
        if room_area < 1000:  # Adjust threshold based on your image size
            print(f"Room {room_id} is too small (area: {room_area}), skipping")
            continue
        
        # Find bounding box
        room_indices = np.where(room_mask_original)
        if len(room_indices[0]) == 0:
            continue
            
        y_indices, x_indices = room_indices
        
        # Get min/max coordinates for bounding box
        try:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Debug output
            print(f"Room {room_id} bbox: ({x_min}, {y_min}, {x_max}, {y_max})")
            
            # Ensure coordinates are valid
            if x_min >= x_max or y_min >= y_max:
                print(f"Invalid bounding box for room {room_id}")
                continue
                
            # Add some padding
            padding = 5
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w - 1, x_max + padding)
            y_max = min(h - 1, y_max + padding)
                
        except ValueError as e:
            print(f"Error finding bounding box for room {room_id}: {e}")
            continue
            
        # Calculate centroid of the room
        centroid_y = int(np.mean(y_indices))
        centroid_x = int(np.mean(x_indices))
        
        # Assign room name based on the closest centroid
        room_name = f"Room_{room_id}"
        min_dist = float('inf')
        
        for label, (label_x, label_y) in room_labels.items():
            dist = np.sqrt((centroid_x - label_x)**2 + (centroid_y - label_y)**2)
            if dist < min_dist:
                min_dist = dist
                room_name = label
        
        # Extract room from original image
        try:
            room_image = original[y_min:y_max+1, x_min:x_max+1].copy()
            
            # Verify that the extracted image is not empty
            if room_image.size == 0 or room_image.shape[0] == 0 or room_image.shape[1] == 0:
                print(f"Extracted empty image for room {room_id}, skipping")
                continue
                
        except Exception as e:
            print(f"Error extracting image for room {room_id}: {e}")
            continue
        
        # Clean room name for filename
        clean_name = ''.join(c if c.isalnum() else '_' for c in room_name)
        room_filename = f"{clean_name}_{room_id}.png"
        room_path = os.path.join(output_dir, room_filename)
        
        # Add text label to the room image
        try:
            cv2.putText(room_image, room_name, (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save room image
            print(f"Saving room {room_id} as {room_name} to {room_path}")
            cv2.imwrite(room_path, room_image)
                
        except Exception as e:
            print(f"Error saving image for room {room_id}: {e}")
            continue
        
        # Store room information
        rooms_info[room_id] = {
            'name': room_name,
            'centroid': (centroid_x, centroid_y),
            'image_path': room_path,
            'bbox': (x_min, y_min, x_max, y_max),
            'area': room_area
        }
        
        # Add text to the colored room visualization
        cv2.putText(colored_rooms_trimmed, room_name, (centroid_x, centroid_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    print(f"Successfully processed {len(rooms_info)} rooms")
    
    # Mark room boundaries on the original
    annotated = original.copy()
    for room_id, info in rooms_info.items():
        x_min, y_min, x_max, y_max = info['bbox']
        color = colors[room_id].tolist()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(annotated, info['name'], info['centroid'], 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imwrite("8_annotated.png", annotated)
    
    return original, colored_rooms_trimmed, rooms_info

def visualize_results(original, segmented, rooms_info):
    """
    Visualize the original image and segmentation results
    """
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Floor Plan")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmented Rooms")
    plt.imshow(segmented)
    plt.axis('off')
    
    # Print room information
    print("\nIdentified Rooms:")
    for room_id, info in rooms_info.items():
        print(f"Room {room_id}: {info['name']} (saved to {os.path.basename(info['image_path'])})")
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your floor plan image path
    base_dir = os.getcwd()
    image_path = os.path.join(base_dir, "resources", "images", "page_2.png")
    output_dir = os.path.join(base_dir, "resources", "processed", "rooms", "segmented2")

    
    try:
        original, segmented, rooms_info = segment_rooms_from_simple_floorplan(image_path, output_dir)
        
        # Export room data to a CSV file
        import csv
        with open('room_data.csv', 'w', newline='') as csvfile:
            fieldnames = ['room_id', 'name', 'centroid_x', 'centroid_y', 'area', 'image_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for room_id, info in rooms_info.items():
                writer.writerow({
                    'room_id': room_id,
                    'name': info['name'],
                    'centroid_x': info['centroid'][0],
                    'centroid_y': info['centroid'][1],
                    'area': info['area'],
                    'image_path': info['image_path']
                })
        
        if rooms_info:
            visualize_results(original, segmented, rooms_info)
            print(f"\nAll room images saved to the '{output_dir}' directory")
        else:
            print("No valid rooms were found in the image")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
   