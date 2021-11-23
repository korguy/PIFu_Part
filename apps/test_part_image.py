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
from lib.data import *
from lib.model import HGPIFuPart
from lib.train_util import *
from lib.mesh_util import *

opt = BaseOptions().parse()

result_path = "./sample_recon"
os.makedirs(result_path, exist_ok=True)

def test(opt):
    cuda = torch.device('cuda')

    dataset = EvalWPoseDataset(opt)

    data_loader = DataLoader(dataset,
                  batch_size=1, shuffle=False,
                  num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    projection_mode = 'orthogonal'
    net = HGPIFuPart(opt, "orthogonal").to(device=cuda)

    if opt.resume_epoch == -1:
        print("Model must be specified.")
        return

    net.load_state_dict(torch.load(os.path.join(opt.load_checkpoints_path,
                                                    opt.name,
                                                    f"net_epoch_{opt.resume_epoch}")))
    def set_eval():
        net.eval()
        net.netF.eval()
        net.netB.eval()

    with torch.no_grad():
        print("Evaluating")
        set_eval()
        grid_ply_generate = True

        for idx, test_data in enumerate(data_loader):
            save_path = '%s/%s/result%02d.obj' % (result_path, opt.name, idx)
            gen_mesh_color(opt.resolution, net, cuda, test_data, save_path, thresh=0.5, use_octree=True, components=False)

if __name__ == "__main__":
    test(opt)