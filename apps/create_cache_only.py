import sys 
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random 
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from lib.options import BaseOptions

opt = BaseOptions().parse()

def create_cache(opt):
    train_dataset = PoseTrainDataset(opt, phase='train')
    test_dataset = PoseTrainDataset(opt, phase='eval')
    for _ in train_dataset:
        pass
    for _ in test_dataset:
        pass

if __name__ == '__main__':
    create_cache(opt)




