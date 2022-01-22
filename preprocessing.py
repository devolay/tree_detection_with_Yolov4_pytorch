from os import strerror
import numpy as np
import glob
from osgeo import gdal
import rasterio
from shared.constants import OPT_LIST


def boost_green(img, scalar):
    boosted_img = (np.dstack((
    (img[:,:,0]).astype(int),
    np.clip((img[:,:,1]*scalar).astype(int), 0, 255),
    (img[:,:,2]).astype(int),
    )))
    return boosted_img

def delete_empty_annotations(path):
    annotations = open(path, "r")
    content = annotations.read()
    lines = content.split("\n")
    new_annotations = ""
    for idx, preds in enumerate(lines):
        pred = preds.split(" ")
        if not len(pred) <= 1:
            new_annotations += lines[idx]
            new_annotations += '\n'
    annotations = open(path, 'w')
    annotations.write(new_annotations)

def merge_images_to_map(dir):
    demList = glob.glob(f'{dir}\\*.tiff')   
    assert len(demList) > 0

    vrt = gdal.BuildVRT(f'{dir}/merged.vrt', demList)
    options_string = " ".join(OPT_LIST)  

    src = rasterio.open(f'{dir}/0.tiff', mode='r+')
    res = (src.res[0], src.res[1])
    src.close()
    gdal.Translate(f'{dir}/merged.png', vrt, options=options_string, xRes = res[0], yRes = res[1])
    gdal.Translate(f'{dir}/merged.tiff', vrt, format='GTiff', xRes = res[0], yRes = res[1])
