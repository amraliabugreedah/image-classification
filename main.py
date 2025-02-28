import os
from src.cv.preprocess import preprocess_image


def main():
    # Define the base directory (assuming you run from the project root)
    base_dir = os.getcwd()

    # Path to the input image in the resources/images folder
    resources_dir = os.path.join(base_dir, "resources", "images")
    image_filename = "page_2.png"  # Replace with your actual file name
    image_path = os.path.join(resources_dir, image_filename)

    # Define an output directory for processed images
    output_dir = os.path.join(base_dir, "resources", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_image_path = os.path.join(output_dir, "preprocessed_page_2.png")

    # Run preprocessing on the input image
    preprocess_image(image_path, output_image_path)

    print("Preprocessing complete. Preprocessed image saved at:", output_image_path)

    # Future steps could be added here:
    # 1. Room detection using OpenCV (from src/cv/plan_detection.py)
    # 2. OCR extraction (from src/ocr/text_extraction.py)
    # 3. LLM/VLM interpretation (from src/llm/analysis.py)
    # 4. Rule checking (from src/rules/check_rules.py)


if __name__ == "__main__":
    main()
