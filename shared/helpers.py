from typing import Union, Any
from pathlib import Path
import yaml
import cv2
import rasterio
import csv
import os, shutil


def load_yaml(filepath: Union[str, Path]) -> Any:
    """Load data from a yaml file."""
    with open(filepath, "r") as f:
        data = yaml.load(f, yaml.FullLoader)
    return data

def check_if_label_inside(potential_image_x: int, potential_image_y: int, label_points, out_image_size: int) -> bool:
    """
    Check if any corner of the bounding box label is inside the cropped image
    """
    return (
        check_if_point_inside(potential_image_x, potential_image_y, label_points[0][0], label_points[0][1], out_image_size)
        or
        check_if_point_inside(potential_image_x, potential_image_y, label_points[0][0], label_points[1][1], out_image_size) 
        or
        check_if_point_inside(potential_image_x, potential_image_y, label_points[1][0], label_points[0][1], out_image_size) 
        or
        check_if_point_inside(potential_image_x, potential_image_y, label_points[1][0], label_points[1][1], out_image_size)
    )

def check_if_point_inside(potential_image_x: int, potential_image_y: int, point_x: float, point_y: float, out_image_size: int) -> bool:
    image_bottom_left = (potential_image_x, potential_image_y)
    image_top_right = (potential_image_x + out_image_size, potential_image_y + out_image_size)
    return (point_x > image_bottom_left[0] and point_x < image_top_right[0] and point_y > image_bottom_left[1] and point_y < image_top_right[1])

def check_if_blank(image: Any, threshold: float) -> bool:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        non_zero_count = cv2.countNonZero(gray_image)
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        return non_zero_count/total_pixels < threshold

def convert_pixel_to_location(x, y, map_path):
    with rasterio.open(map_path) as map_layer:
        pixels2coords = map_layer.xy(x,y)
    return pixels2coords

def convert_location_to_pixels(lat, lng, map_path):
    with rasterio.open(map_path) as map_layer:
        coords2pixels = map_layer.index(lng, lat)
    return coords2pixels

def save_coords_list_as_csv(coords, path):
    with open(path) as f:
        write = csv.writer(f)
        write.writerow(['Latitude', 'Longitude', 'Degree Width', 'Degree Height'])
        write.writerows(coords)

def clear_temp_directory(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete temporary files %s. Reason: %s' % (file_path, e))