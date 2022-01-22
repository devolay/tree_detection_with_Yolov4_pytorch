import argparse
import pandas as pd
import glob
import rasterio
import os

from easydict import EasyDict as edict
from owslib.wms import WebMapService
from shared.constants import (
    LAYER,
    STYLE,
    SRS,
    IMAGE_FORMAT,
    PROJ4
)


class ImageFetcher():
    
    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config

    def download_data_from_coords(
        self,
        min_lat: float, 
        min_lng: float, 
        max_lat: float, 
        max_lng: float, 
        output_dir: str
    ):
        data = pd.DataFrame()
        lat = min_lat
        lng = min_lng
        index = 0
        while(lat <= max_lat and lng <= max_lng): 
            row = {"ID": index, "lat": lat, "lng": lng}
            data = data.append(row, ignore_index=True)
            index += 1
            lat += (self.pipeline_config.lat_value * 2)
            if lat > max_lat:
                lng += (self.pipeline_config.long_value * 2)
                lat = min_lat
        self.download_data_from_df(data, output_dir)


    def download_data_from_df(
        self,
        data: pd.DataFrame, 
        output_dir: str
    ):
        wms = WebMapService(self.pipeline_config.geoportal_endpoint, version='1.1.1')
        for _, row in data.iterrows():
            img = wms.getmap(
            layers=[LAYER],
            styles=[STYLE],
            srs=SRS,
            bbox=(
                row.lng - self.pipeline_config.long_value, 
                row.lat - self.pipeline_config.lat_value, 
                row.lng + self.pipeline_config.long_value, 
                row.lat + self.pipeline_config.lat_value
            ),
            size=(self.pipeline_config.img_height, self.pipeline_config.img_width),
            format=IMAGE_FORMAT
        )
            path = f'{output_dir}\{int(row.ID)}.tiff'
            out = open(path, 'wb')
            out.write(img.read())
            out.close()
            self._create_metadata(path, row)


    def _create_metadata(self, image_path, row):
        dataset = rasterio.open(image_path, 'r+')
        bands = [1, 2, 3]
        data = dataset.read(bands)
        _, width, height = data.shape
        transform = rasterio.transform.from_origin(
            row.lng - self.pipeline_config.long_value, 
            row.lat + self.pipeline_config.lat_value, 
            self.pipeline_config.long_value * 2 / width, 
            self.pipeline_config.lat_value * 2 / height
            )
        crs = rasterio.crs.CRS.from_string(PROJ4)
        dataset.close()
        with rasterio.open(
            image_path, 'w+', driver='GTiff',
            width=width, height=height,
            count=3, dtype=data.dtype, nodata=0,
            transform=transform, crs=crs
            ) as dst:
            dst.write(data, indexes=bands)
            dst.close()
        print(str(int(row.ID)) + " image with metadata created")

