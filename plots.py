import os
import random
from PIL import Image
from collections import Counter
from tqdm import tqdm
import json


def return_task_grid(task_info: str) -> dict:
    training_jsons = os.listdir("dataset/training")
    testing_jsons = os.listdir("dataset/evaluation/")
    # task_json = task_info + ".json"

    if ".json" not in task_info:
        task_json = task_info + ".json"
    else:
        task_json = task_info

    if task_json in training_jsons:
        json_file_path = os.path.join("dataset/training", task_json)
    elif task_json in testing_jsons:
        json_file_path = os.path.join("dataset/evaluation", task_json)
    else:
        raise Exception("No such task found!")

    with open(json_file_path) as file:
        task = json.load(file)
    return task


# Convert hex color to RGB
def hex_to_rgb(hex):
    return tuple(int(hex[i : i + 2], 16) for i in (1, 3, 5))


# Define the color map in both hex and RGB
color_map = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
color_map_rgb = [hex_to_rgb(color) for color in color_map]


# Function to get the next file number in the folder
def get_next_file_number(folder: str) -> int:
    existing_files = [int(f.split(".")[0]) for f in os.listdir(folder) if f.split(".")[0].isdigit()]
    return max(existing_files, default=0) + 1 if existing_files else 1


def process_images(input_folder="consolidated_images", output_folder="temp"):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            img_data = list(img.getdata())

            # Count the frequency of each color
            color_count = Counter(img_data)

            # Identify the most and least frequent colors (excluding black)
            most_common_color = None
            least_common_color = None

            # Find the most common color that is not black
            for color, count in color_count.most_common():
                if color != (0, 0, 0):
                    most_common_color = color
                    break

            # Find the least common color that is not black
            for color, count in color_count.most_common()[::-1]:
                if color != (0, 0, 0):
                    least_common_color = color
                    break

            # Process image for most common color
            if most_common_color:
                replacement_colors = [
                    color for color in color_map_rgb if color not in [(0, 0, 0), most_common_color]
                ]
                replacement_color = random.choice(replacement_colors)
                new_img_data = [
                    replacement_color if pixel == most_common_color else pixel for pixel in img_data
                ]
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(new_img_data)
                next_file_number = get_next_file_number(output_folder)
                output_path = os.path.join(output_folder, f"{next_file_number}.png")
                new_img.save(output_path)

            # Process image for least common color
            if least_common_color:
                replacement_colors = [
                    color for color in color_map_rgb if color not in [(0, 0, 0), least_common_color]
                ]
                replacement_color = random.choice(replacement_colors)
                new_img_data = [
                    replacement_color if pixel == least_common_color else pixel
                    for pixel in img_data
                ]
                new_img = Image.new(img.mode, img.size)
                new_img.putdata(new_img_data)
                next_file_number = get_next_file_number(output_folder)
                output_path = os.path.join(output_folder, f"{next_file_number}.png")
                new_img.save(output_path)


if __name__ == "__main__":
    process_images()
