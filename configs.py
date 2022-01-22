from dataclasses import dataclass
from dacite import Config
import enum
import dacite

@dataclass(repr=False)
class PipelineConfig:
    pipeline_type: str
    weightfile: str
    geoportal_endpoint: str
    img_height: int
    img_width: int
    lat_value: float
    long_value: float
    temp_files_output_dir: str
    use_cuda: bool
    inference_output_dir: str
    conf_threshold: float
    nms_threshold: float

    @classmethod
    def from_dict(cls, input_dict: dict) -> "PipelineConfig":
        return dacite.from_dict(cls, input_dict, config=Config(cast=[enum.Enum]))