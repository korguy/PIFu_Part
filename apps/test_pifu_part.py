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
summary_path = os.path.join('.', 'runs', f'{opt.name}')
os.makedirs(summary_path, exist_ok=True)

def test(opt):
    cuda = torch.device('cuda')
    test_dataset = PoseTrainDataset(opt, phase='eval')
    projection_mode = test_dataset.projection_mode
    
    test_data_loader = DataLoader(test_dataset,
                                batch_size=1, shuffle=True,
                                num_workers=0, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))
    
    net = HGPIFuPart(opt, "orthogonal").to(device=cuda)
    print("Using Network: ", net.name)
    net.load_FB()
    
    if opt.resume_epoch != -1:
        print(f"resuming from epoch {opt.resume_epoch}")
        net.load_state_dict(torch.load(os.path.join(opt.load_checkpoints_path,
                                                    opt.name,
                                                    f"net_epoch_{opt.resume_epoch}")))
    else:
        raise Exception('resume epoch must be specified')
        
    def set_eval():
        net.eval()
        net.mlp.eval()
        
    os.makedirs(opt.load_checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs(os.path.join(opt.load_checkpoints_path, opt.name), exist_ok=True)
    os.makedirs(os.path.join(opt.results_path, opt.name), exist_ok=True)
    
    print("Evaluating...")
    with torch.no_grad():
        set_eval()
        err_part_arr, err_occ_arr, IOU_arr, prec_arr, recall_arr = [], [], [], [], []
        rnd_sub = np.random.randint(0, high=len(test_data_loader))
        for idx, test_data in enumerate(test_data_loader, start=rnd_sub):
            img_tensor = test_data['img'].to(device=cuda)
            calib_tensor = test_data['calib'].to(device=cuda)
            samples_tensor = test_data['samples'].to(device=cuda)
            labels_tensor = test_data['labels'].to(device=cuda)
            parts_tensor = test_data['parts'].to(device=cuda)

            res, _error, part = net.forward(img_tensor, samples_tensor, calib_tensor, labels_tensor, parts_tensor)

            IOU, prec, recall = compute_acc(res, labels_tensor)

            err_part_arr.append(_error['Err(part)'].item())
            err_occ_arr.append(_error['Err(occ)'].item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())
            save_path = '%s/%s/recon/result_%d.obj' % (opt.results_path, opt.name, opt.resume_epoch)
            gen_mesh(opt.resolution, net, cuda, test_data, save_path, components=None)
            break
        eval_errors = np.average(err_part_arr), np.average(err_occ_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)
        print('eval test err(part): {0:06f} | err(occ): {0:06f} | IOU: {1:06f} | prec: {2:06f} | recall: {3:06f}'.format(*eval_errors))    
        save_path = '%s/%s/pred%d.ply' % (opt.results_path, opt.name, opt.resume_epoch)
        save_gt_path = '%s/%s/pred_gt%d.ply' % (opt.results_path, opt.name, opt.resume_epoch)
        save_part_path = '%s/%s/part%d.ply' % (opt.results_path, opt.name, opt.resume_epoch)
        save_gt_part = '%s/%s/part_gt%d.ply' % (opt.results_path, opt.name, opt.resume_epoch)
        r = res[0].cpu()
        points = samples_tensor[0].transpose(0, 1).cpu()

        save_samples_truncated_prob(save_gt_path, points.detach().numpy(), labels_tensor[0].cpu().detach().numpy())
        save_samples_truncated_part(save_gt_part, points.detach().numpy(), parts_tensor[0].cpu().detach().numpy())
        save_samples_truncated_part(save_part_path, points.detach().numpy(), part[0].cpu().numpy())
        save_samples_truncated_prob(save_path, points.detach().numpy(), r.detach().numpy())
    
if __name__ == '__main__':
    test(opt)