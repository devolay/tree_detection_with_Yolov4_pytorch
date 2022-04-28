# Tree detection based on YOLOv4

Link to github repository https://github.com/devolay/tree_detection_with_Yolov4_pytorch

The repository uses https://github.com/roboflow-ai/pytorch-YOLOv4 repository which is based on Apache 2.0 licenses

Download the best trained model from https://drive.google.com/file/d/1LyEyLLPQy7sXEk9fLnuBUV4-LX0C62NJ/view?usp=sharing and point to it in the pipeline_config.yaml file

## How to run detection?

1. Create virtual envirionment and install packages from requirements.txt

```
python3 -m venv /path/to/new/virtual/environment
```

```
source /path/to/new/virtual/environment
````

```
pip install -r requirements.txt
```

2. Install gdal separately from https://gdal.org/download.html

3. Copy the YOLOv4 model adapted to tree detection from https://github.com/devolay/yolov4_tree_edit and put it to the directory model/yolov4

4. Now you can prepare configuration file's fields:

`temp_files_output_dir` - Absolute path to the directory where temporary files will be saved. (After the end of detection, this folder will be emptied.

`weightfile` - Absolute path to the model weights which you can download from https://drive.google.com/file/d/1LyEyLLPQy7sXEk9fLnuBUV4-LX0C62NJ/view?usp=sharing

`inference_output_dir` - Absolute path to directory where output of detection should be saved.

I do not recommend changing the other configuration parameters.

`WARNING!` A graphics card is required to run the model
