from copy import deepcopy
import os
import numpy as np
import base64
import json
import cv2
import random

from typing import Tuple
from shared.helpers import check_if_blank, check_if_label_inside, convert_pixel_to_location
from typing import Any, List, Dict
from configs import PipelineConfig
from preprocessing import boost_green




class DatasetGenerator():

    def __init__(self, pipeline_config: PipelineConfig, coords: Tuple[float]):
        self.pipeline_config = pipeline_config
        self.coords = coords

        temp_path = self.pipeline_config.temp_files_output_dir
        if self.pipeline_config.pipeline_type == "inference":
            self.gen_type = "mesh"
        else:
            self.gen_type = "random"
            self.samples_count = 500
            self.labels_json_path = temp_path + "merged.json"

        self.out_images_size = 608
        self.map_image_path = temp_path + "/merged.png"
        self.dataset_path = temp_path + "/dataset"
        

    def generate(self):
        image = cv2.imread(self.map_image_path, cv2.IMREAD_UNCHANGED)
        max_random_x = image.shape[1] - self.out_images_size
        max_random_y = image.shape[0] - self.out_images_size
        if self.gen_type == 'random':
            return self._random_generate(image, max_random_x, max_random_y)
        elif self.gen_type == 'mesh':
            return self._inference_mesh_generate(image, max_random_x, max_random_y)
    
    def _random_generate(self, image: np.ndarray, max_random_x: int, max_random_y: int):
        json_file = open(self.labels_json_path)
        map_labels = json.load(json_file)
        images_count = 0
        while images_count != self.samples_count:
            x = random.randint(0, max_random_x)
            y = random.randint(0, max_random_y)
            potential_image = image[y:y+self.out_images_size, x:x+self.out_images_size]
            if check_if_blank(potential_image, 0.80):
                continue
            new_labels = self._generate_new_labels(map_labels, x, y)
            if len(new_labels) > 0:
                image_path = self.pipeline_config.temp_files_output_dir + "/" + str(images_count) + ".jpg"
                img = boost_green(potential_image, 1.2)
                cv2.imwrite(image_path, img)
                self._generate_new_json(images_count, new_labels, potential_image)
                images_count += 1   

    #Generating only images from map (without jsons labels) - for inference
    def _inference_mesh_generate(self, image: np.ndarray, max_random_x: int, max_random_y: int):
        x = 0
        y = 0
        images_count = 0
        relative_lat = self.coords[2] - self.coords[0]
        relative_long = self.coords[3] - self.coords[1]
        actual_img_coords = {}
        actual_img_pixels = {}
        isExist = os.path.exists(self.dataset_path)
        if not isExist:
            os.makedirs(self.dataset_path)
            print("Dataset directory is created!")

        while x <= max_random_x and y <= max_random_y:
            potential_image = image[y:y+self.out_images_size, x:x+self.out_images_size]
            if check_if_blank(potential_image, 0.80):
                x += 304
                if x > max_random_x:
                    x = 0
                    y += 304
                continue    

            image_path = self.dataset_path + "/" + str(images_count) + ".jpg"
            img = boost_green(potential_image, 1.2)
            cv2.imwrite(image_path, img)

            min_lat, min_lng= convert_pixel_to_location(x, y, self.pipeline_config.temp_files_output_dir + "/merged.tiff")
            max_lat, max_lng = convert_pixel_to_location(
                x+self.out_images_size, 
                y+self.out_images_size, 
                self.pipeline_config.temp_files_output_dir + "/merged.tiff"
            )

            actual_img_coords[images_count] = [min_lat, min_lng, max_lat, max_lng]
            actual_img_pixels[images_count] = [x, y, x+self.out_images_size, y+self.out_images_size]
            images_count += 1
            x += 304
            if x > max_random_x:
                x = 0
                y += 304

        return actual_img_coords, actual_img_pixels
    
    def _generate_new_labels(
        self,
        labels: List[Dict[str,Any]], 
        potential_image_x: float, 
        potential_image_y: float
    ) -> List[Dict[str,Any]]:

        assert labels["shapes"] is not None
        image_labels = []
        labels_copy = deepcopy(labels["shapes"])
        for label in labels_copy:
            label_points=label["points"]
            if check_if_label_inside(potential_image_x, potential_image_y, label_points, self.out_images_size):
                points_min = label["points"][0]
                points_max = label["points"][1]
                original_area = (points_max[0] - points_min[0]) * (points_max[1] - points_min[1])
                label["points"][0][0] -= potential_image_x
                label["points"][0][1] -= potential_image_y
                label["points"][1][0] -= potential_image_x
                label["points"][1][1] -= potential_image_y
                if points_min[0] < 0:
                    label["points"][0][0] = 0
                if points_min[1] < 0:
                    label["points"][0][1] = 0
                if points_max[0] > self.out_images_size:
                    label["points"][1][0] = self.out_images_size
                if points_max[1] > self.out_images_size:
                    label["points"][1][1] = self.out_images_size
                new_area = (label["points"][1][0] - label["points"][0][0]) * (label["points"][1][1] - label["points"][0][1])
                if new_area / original_area > 0.20:
                    image_labels.append(label)
        return image_labels

    def _generate_new_json(self, images_count, labels: List[Dict[str,Any]], potential_image):
        label_dir = self.pipeline_config.temp_files_output_dir + "/" + str(images_count) + ".json"
        img = boost_green(potential_image, 1.2)
        _, imdata = cv2.imencode('.PNG', img)
        image_string = base64.b64encode(imdata).decode('ascii')
        labels_json ={
        "version": "4.6.0",
        "flags": {},
        "shapes": labels,
        "imagePath": str(images_count) + ".png",
        "imageData": image_string,
        "imageHeight": self.out_images_size,
        "imageWidth": self.out_images_size,
        }
        with open(label_dir, 'w') as json_file:
            json.dump(labels_json, json_file)
