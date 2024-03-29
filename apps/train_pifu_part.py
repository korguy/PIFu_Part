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
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

from lib.options import BaseOptions
from lib.data import *
from lib.model import HGPIFuPart
from lib.train_util import *
from lib.mesh_util import *

opt = BaseOptions().parse()
summary_path = os.path.join('.', 'runs', f'{opt.name}')
os.makedirs(summary_path, exist_ok=True)
writer = SummaryWriter(summary_path)

def sum_dict(los):
    temp = 0
    for l in los:
        temp += los[l]
    return temp

def train(opt):
    cuda = torch.device('cuda')

    train_dataset = PoseTrainDataset(opt, phase='train')
    test_dataset = PoseTrainDataset(opt, phase='eval')
    projection_mode = train_dataset.projection_mode

    train_data_loader = DataLoader(train_dataset,
                                    batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=0, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))
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
    optimizer = torch.optim.RMSprop(net.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def set_train():
        net.train()

    def set_eval():
        net.eval()

    os.makedirs(opt.load_checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs(os.path.join(opt.load_checkpoints_path, opt.name), exist_ok=True)
    os.makedirs(os.path.join(opt.results_path, opt.name), exist_ok=True)

    start_epoch = 0 if opt.resume_epoch == -1 else max(opt.resume_epoch + 1, 0)
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        set_train()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve data
            img_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            samples_tensor = train_data['samples'].to(device=cuda)

            labels_tensor = train_data['labels'].to(device=cuda)
            parts_tensor = train_data['parts'].to(device=cuda)

            res, _error, part = net.forward(img_tensor, samples_tensor, calib_tensor, labels_tensor, parts_tensor)

            optimizer.zero_grad()
            error = sum_dict(_error)
            error.backward()
            optimizer.step()

            if train_idx % opt.freq_plot == 0:
                print(
                    f"Name: {opt.name} | Epoch: {epoch} | {train_idx}/{len(train_data_loader)} | Error['part']: {_error['Err(part)'].item():.06f} | Error['occ']: {_error['Err(occ)'].item():.06f} | LR: {scheduler.get_last_lr()[0]:.05f} | Sigma: {opt.sigma:.02f}")
                writer.add_scalar('training loss',
                                    error.item(),
                                    epoch * len(train_data_loader) + train_idx)

            if train_idx % opt.freq_eval == 0 and train_idx != 0:
                print("Evaluating...")
                with torch.no_grad():
                    set_eval()
                    err_arr , IOU_arr, prec_arr, recall_arr = [], [], [], []
                    for idx, test_data in enumerate(test_data_loader):
                        img_tensor = test_data['img'].to(device=cuda)
                        calib_tensor = test_data['calib'].to(device=cuda)

                        samples_tensor = test_data['samples'].to(device=cuda)
                        parts_tensor = test_data['parts'].to(device=cuda)

                        if len(parts_tensor.shape) == 3:
                            parts_tensor = parts_tensor.view(parts_tensor.shape[0], parts_tensor.shape[2])

                        labels_tensor = test_data['labels'].to(device=cuda)

                        res_eval, eval_err, eval_part = net.forward(img_tensor, samples_tensor, calib_tensor, labels_tensor, parts_tensor)

                        IOU, prec, recall = compute_acc(res_eval, labels_tensor)

                        err_arr.append(sum_dict(eval_err).item())
                        IOU_arr.append(IOU.item())
                        prec_arr.append(prec.item())
                        recall_arr.append(recall.item())

                        if idx == 200:
                            break

                    eval_errors = np.average(err_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)
                    print('eval test err: {0:06f} | IOU: {1:06f} | prec: {2:06f} | recall: {3:06f}'.format(*eval_errors))
                    writer.add_scalar('test loss', eval_errors[0], epoch * len(train_data_loader) + train_idx)
                    writer.add_scalar('test IOU', eval_errors[1], epoch * len(train_data_loader) + train_idx)
                    writer.add_scalar('test precision', eval_errors[2], epoch * len(train_data_loader) + train_idx)
                    writer.add_scalar('test recall', eval_errors[3], epoch * len(train_data_loader) + train_idx) 
                set_train()

            iter_data_time = time.time()

            if train_idx % opt.freq_save_ply == 0 and train_idx != 0:
                save_path = os.path.join(opt.results_path, opt.name, f'pred_{epoch}_{train_idx}.ply')
                save_path2 = os.path.join(opt.results_path, opt.name, f'in_pred_{epoch}_{train_idx}.ply')
                r = res[0].cpu()
                points = samples_tensor[0].transpose(0, 1).cpu()
                save_samples_truncated_prob(save_path2, points.detach().numpy(), labels_tensor[0].cpu().detach().numpy())
                save_samples_truncated_prob(save_path, points.detach().numpy(), r.detach().numpy())
                
            if train_idx % opt.freq_mesh == 0:
                with torch.no_grad():
                    set_eval()
                    for idx, test_data in enumerate(test_data_loader):
                        save_path = '%s/%s/recon/result_%d_%d.obj' % (opt.results_path, opt.name, epoch, train_idx)
                        gen_mesh_color(opt.resolution, net, cuda, test_data, save_path, components=None)
                        break
                set_train()
                
            # for fast debugging
            if train_idx > 10000:
                break
            
        torch.save(net.state_dict(), os.path.join(opt.load_checkpoints_path, opt.name, 'net_latest'))
        torch.save(net.state_dict(), os.path.join(opt.load_checkpoints_path, opt.name, f'net_epoch_{epoch}'))
        scheduler.step()

if __name__ == '__main__':
    train(opt)




