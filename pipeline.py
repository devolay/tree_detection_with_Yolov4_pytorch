from turtle import width
import dacite
import torch
import os
import shutil

from PIL import Image
from configs import PipelineConfig
from dataset_generator import DatasetGenerator
from image_fetcher import ImageFetcher
from shared.helpers import load_yaml, save_coords_list_as_csv, clear_temp_directory
from preprocessing import merge_images_to_map
from model.yolov4.models import Yolov4
from model.yolov4.tool.utils import *
from shared.constants import CLASS_NAMES


class Pipeline():

    def __init__(self, pipeline_config: PipelineConfig):
        self.pipeline_config = pipeline_config

    @classmethod
    def from_dict(cls, pipeline_config_dict: dict) -> "Pipeline":
        pipeline_config = dacite.from_dict(PipelineConfig, pipeline_config_dict)
        return cls(pipeline_config=pipeline_config)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Pipeline":
        pipeline_config = load_yaml(filepath=yaml_path)
        return cls.from_dict(pipeline_config)

    def run_inference(
            self, 
            min_lat: float,
            min_lng: float,
            max_lat: float,
            max_lng: float
        ):
        coords = (min_lat, min_lng, max_lat, max_lng)
        image_fetcher = ImageFetcher(self.pipeline_config)
        image_fetcher.download_data_from_coords(
            coords[0], coords[1], coords[2], coords[3], self.pipeline_config.temp_files_output_dir
        )

        merge_images_to_map(self.pipeline_config.temp_files_output_dir)
        data_generator = DatasetGenerator(self.pipeline_config, coords)
        img_coords, img_pixels = data_generator.generate()

        model = Yolov4(n_classes=1)
        pretrained_dict = torch.load(self.pipeline_config.weightfile, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)
        if self.pipeline_config.use_cuda:
            model.cuda()
        
        actual_box_coords = {}
        dataset_path = self.pipeline_config.temp_files_output_dir + "/dataset"
        images = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
        for imgfile in images:
            img = Image.open(dataset_path + "/" + imgfile).convert('RGB')
            boxes = do_detect(model, img, 0.25, 1, 0.4, self.pipeline_config.use_cuda)
            actual_box_coords[int(imgfile.split('.')[0])] = boxes
        
        self._postprocess_image(img_pixels, actual_box_coords)
        bbox_coords = self._postprocess_coords(img_coords, actual_box_coords)
        clear_temp_directory(self.pipeline_config.temp_files_output_dir)
        return bbox_coords

    def _postprocess_image(self, img_pixels, acutal_box_coords):
        image = Image.open(self.pipeline_config.temp_files_output_dir + "/merged.png").convert('RGB')
        map_width, map_height = image.size
        map_boxes = []
        for img, boxes in acutal_box_coords.items():
            img_corners = img_pixels[img]
            img_height = img_corners[3] - img_corners[1]
            img_width = img_corners[2] - img_corners[0]
            for box in boxes:
                box_img_x = img_corners[0] + (box[0]*img_width)
                box_img_y = img_corners[1] + (box[1]*img_height)
                box_img_width = box[2] * img_width
                box_img_height = box[3] * img_height
                box_map_x = box_img_x / map_width
                box_map_y = box_img_y / map_height
                box_map_width = box_img_width / map_width
                box_map_height = box_img_height / map_height
                new_box = [
                    box_map_x,
                    box_map_y,
                    box_map_width,
                    box_map_height,
                    box[4],
                    box[5]
                ]
                map_boxes.append(new_box)
        map_boxes = nms(map_boxes, self.pipeline_config.nms_threshold)
        map_boxes = [box for box in map_boxes if box[4]> self.pipeline_config.conf_threshold]
        output_path = self.pipeline_config.inference_output_dir + "/output_image.jpg"
        plot_boxes(image, map_boxes, output_path, CLASS_NAMES)

    def _postprocess_coords(self, img_coords, actual_box_coords):
        detected_tree_centers = []
        for img, boxes in actual_box_coords.items():
            img_corners = img_coords[img]
            img_degree_width = img_corners[2] - img_corners[0]
            img_dergee_height = img_corners[3] - img_corners[1]
            for box in boxes:
                long = img_corners[0] + (box[0] * img_degree_width)
                lat = img_corners[1] + (box[1] * img_dergee_height)
                width = box[2] * img_degree_width
                height = box[3] * img_degree_width
                box_degrees = [lat, long, width, height, box[4], box[5]]
                detected_tree_centers.append(box_degrees)
        detected_tree_centers = nms(detected_tree_centers, self.pipeline_config.nms_threshold)
        detected_tree_centers = [[box[0], box[1], box[2], box[3], box[4]] for box in detected_tree_centers]
        np.savetxt(
            self.pipeline_config.inference_output_dir + "/coords.csv", 
            detected_tree_centers,
            delimiter =", ", 
            fmt ='% s'
        )
        return detected_tree_centers