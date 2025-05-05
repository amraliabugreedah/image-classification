import os
from pdf2image import convert_from_path
import time


def pdf_to_images(
    pdf_path, dpi=200, output_folder=None, crop_margin=0, margin_in_percent=False
):
    """
    Convert a PDF file to high-quality PNG images.

    Args:
        pdf_path (str): The path to the input PDF.
        dpi (int): Dots per inch for conversion (higher means better quality).
        output_folder (str): Directory to save the images. If None, images are not saved.
        crop_margin (int or tuple): Margin(s) to crop from each edge.

    Returns:
        list: A list of file paths to the generated images.
    """
    # Convert PDF pages to images using pdf2image
    images = convert_from_path(
        pdf_path, dpi=100, fmt="png", thread_count=os.cpu_count(), use_pdftocairo=True
    )
    output_paths = []

    # Save each image if an output folder is provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Determine crop margins based on whether we're using percentages or pixels
    if margin_in_percent:
        if isinstance(crop_margin, (int, float)):
            left_pct = top_pct = right_pct = bottom_pct = float(crop_margin)
        elif isinstance(crop_margin, (tuple, list)) and len(crop_margin) == 4:
            left_pct, top_pct, right_pct, bottom_pct = map(float, crop_margin)
        else:
            raise ValueError(
                "crop_margin must be a float or a tuple/list of four floats when margin_in_percent is True"
            )
    else:
        # Determine crop margins
        if isinstance(crop_margin, (int, float)):
            left_margin = top_margin = right_margin = bottom_margin = int(crop_margin)
        elif isinstance(crop_margin, (tuple, list)) and len(crop_margin) == 4:
            left_margin, top_margin, right_margin, bottom_margin = map(int, crop_margin)
        else:
            raise ValueError("crop_margin must be an int or a tuple/list of four ints")

    for i, img in enumerate(images):
        filename = f"page_{i+1}.png"

        # Get original image dimensions
        img_width, img_height = img.size  # PIL.Image.size returns (width, height)

        if margin_in_percent:
            # Calculate pixel margins from percentages
            left_margin = int(img_width * left_pct)
            top_margin = int(img_height * top_pct)
            right_margin = int(img_width * right_pct)
            bottom_margin = int(img_height * bottom_pct)

        # Ensure margins don't exceed image dimensions
        left = min(left_margin, img_width // 2)
        top = min(top_margin, img_height // 2)
        right = max(img_width - right_margin, left + 1)
        bottom = max(img_height - bottom_margin, top + 1)

        crop_box = (left, top, right, bottom)
        cropped_img = img.crop(crop_box)

        if output_folder:
            filepath = os.path.join(output_folder, filename)
            cropped_img.save(filepath, "PNG")
            output_paths.append(filepath)
        else:
            # If not saving to disk, you might return the image objects directly
            output_paths.append(cropped_img)

    return output_paths


if __name__ == "__main__":
    # Example usage
    pdf_path = os.path.join(os.getcwd(), "resources", "pdfs", "sample_plan.pdf")
    output_folder = os.path.join(os.getcwd(), "resources", "images", "pdf-to-images")

    start_time = time.time()

    # Use different percentage margins for each side (left, top, right, bottom)
    images_tuple = pdf_to_images(
        pdf_path,
        dpi=300,
        output_folder=output_folder,
        crop_margin=0,
        # crop_margin=(0.023, 0.023, 0.22, 0.023),
        margin_in_percent=True,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Converted pages:", images_tuple)
    print(f"Time taken: {elapsed_time:.2f} seconds")
