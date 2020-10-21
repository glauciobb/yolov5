"""Convert pretrained YOLOv5 *.pt model to state_dict format

Usage:
    $ python models/convert.py --weights ./weights/yolov5s.pt
"""

import argparse
import sys
import time
from pathlib import Path
import torch

sys.path.append('./')  # to run '$ python *.py' files in subdirectories


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../weights/yolov5s.pt', help='weights path')
    opt = parser.parse_args()
    t = time.time()

    # Load PyTorch model
    model = torch.load(opt.weights, map_location='cpu')['model']

    ckpt = {'names': model.names,
            'yaml': model.yaml,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}

    name = Path(opt.weights).stem

    torch.save(ckpt, f'../weights/{name}_sd.pt')

