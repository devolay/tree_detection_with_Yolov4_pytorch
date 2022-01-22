from pipeline import Pipeline
import argparse
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default=None, help='Path to config .yaml file')
parser.add_argument('--min_lat', type=float, default=None, help='Minimum latitude value for detection area')
parser.add_argument('--min_lng', type=float, default=None, help='Minimum longitude value for detection area')
parser.add_argument('--max_lat', type=float, default=None, help='Maximum latitude value for detection area')
parser.add_argument('--max_lng', type=float, default=None, help='Maximum longitude value for detection area')
args = parser.parse_args()

def main():
    pipeline = Pipeline.from_yaml(args.config)
    bbox_coords = pipeline.run_inference(args.min_lat, args.min_lng, args.max_lat, args.max_lng)
    print("Inference work completed!")
    pprint(bbox_coords)

if __name__ == "__main__":
    main()
