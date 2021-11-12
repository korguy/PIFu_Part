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

opt = BaseOptions().parse()
summary_path = os.path.join('.', 'runs', f'{opt.name}')
os.makedirs(summary_path, exist_ok=True)
writer = SummaryWriter(summary_path)

def train(opt):
    cuda = torch.device(f'cuda:{opt.gpu_id}')

    train_dataset = PoseTrainDataset(opt, phase='train')
    test_dataset = PoseTrainDataset(opt, phase='eval')
    projection_mode = train_dataset.projection_mode

    train_data_loader = DataLoader(train_dataset,
                                    batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                    num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('train data size: ', len(train_data_loader))
    test_data_loader = DataLoader(test_dataset,
                                    batch_size=1, shuffle=True,
                                    num_workers=opt.num_threads, pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))
    
    for tmp in test_data_loader:
        for key in tmp.keys():
            try:
                print(key, tmp[key].shape)
            except:
                print(key, tmp[key])

    net = HGPIFuPart(opt, "orthogonal").to(device=cuda)
    print("Using Network: ", net.name)
    net.load_FB()

    if opt.resume_epoch != -1:
        print(f"resuming from epoch {opt.resume_epoch}")
        net.load_state_dict(torch.load(os.path.join(opt.checkpoints_path,
                                                    opt.name,
                                                    f"net_epoch_{opt.resume_epoch}")))
    optimzer = torch.optim.RMSprop(net.parameters(), lr=opt.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for _ in range(opt.resume_epoch):
        scheduler.step()

    def set_train():
        net.train()

    def set_eval():
        net.eval()

# 	os.makedirs(opt.checkpoints_path, exist_ok=True)
# 	os.makedirs(opt.results_path, exist_ok=True)
# 	os.makedirs(os.path.join(opt.checkpoints_path, opt.name), exist_ok=True)
# 	os.makedirs(os.path.join(opt.results_path, opt.name), exist_ok=True)

# 	start_epoch = 0 if opt.resume_epoch == -1 else max(opt.resume_epoch, 0)
# 	for epoch in range(start_epoch, opt.num_epoch):
# 		epoch_start_time = time.time()
# 		iter_data_time = time.time()
# 		for train_idx, train_data in enumerate(train_data_loader):
# 			set_train()
# 			iter_start_time = time.time()

# 			# retrieve data
# 			img_tensor = train_data['img'].to(device=cuda)
# 			calib_tensor = train_data['calib'].to(device=cuda)
# 			samples_tensor = train_data['samples'].to(device=cuda)

# 			labels_tensor = train_data['labels'].to(device=cuda)
# 			parts_tensor = train_data['parts'].to(device=cuda)

# 			res, error = net.forward(img_tensor, samples_tensor, calib_tensor, labels_tensor, parts_tensor)

# 			optimzer.zero_grad()
# 			error['Err(part)'].backward(retain_graph=True)
# 			error['Err(occ)'].backward()
# 			optimzer.step()

# 			if train_idx % opt.freq_plot == 0:
# 				print(
# 					f"Name: {opt.name} | Epoch: {epoch} | {train_idx}/{len(train_data_loader)} | \
# 					Error(part): {error['Err(part)'].item():.06f} | Error(occ): {error['Err(occ)]'].item():.06f} \
# 					| LR: {scheduler.get_last_lr()[0]:.05f} | Sigma: {opt.sigma:.02f}")
# 				writer.add_scalar('training loss (part)',
# 									error['Err(part)'].item(),
# 									epoch * len(train_data_loader) + train_idx)
# 				writer.add_scalar('training loss (occ)',
# 									error['Err(occ)'].item(),
# 									epoch * len(train_data_loader) + train_idx)

# 			if train_idx % opt.freq_eval == 0:
# 				print("Evaluating...")
# 				set_eval()
# 				occ_error_arr, part_error_arr, IOU_arr, prec_arr, recall_arr = [], [], [], [], []
# 				for idx, test_data in enumerate(test_data_loader):
# 					img_tensor = test_data['img'].to(device=cuda)
# 					calib_tensor = test_data['calib'].to(device=cuda)

# 					samples_tensor = test_data['samples'].to(device=cuda)
# 					parts_tensor = test_data['parts'].to(device=cuda)

# 					res_eval, _ = net.forward(img_tensor, samples_tensor, calib_tensor, labels_tensor, parts_tensor)

# 					IOU, prec, recall = compute_acc(res_eval, labels_tensor)

# 					occ_error_arr.append(error['Err(occ)'].item())
# 					part_error_arr.append(error['Err(part)'].item())
# 					IOU_arr.append(IOU.item())
# 					prec_arr.append(prec.item())
# 					recall_arr.append(recall.item())

# 				eval_errors = np.average(occ_error_arr), np.average(part_error_arr), np.average(IOU_arr), np.average(prec_arr), np.average(reacll_arr)
# 				print('eval test err(occ): {0:06f} | err(part): {1:06f} | IOU: {2:06f} | prec: {3:06f} | recall: {4:06f}'.format(*eval_errors))
# 				writer.add_scalar('test loss(occ)', eval_errors[0], epoch * len(train_data_loader) + train_idx)
# 				writer.add_scalar('test loss(part)', eval_errors[1], epoch * len(train_data_loader) + train_idx)
# 				writer.add_scalar('test IOU', eval_errors[2], epoch * len(train_data_loader) + train_idx)
# 				writer.add_scalar('test precision', eval_errors[3], epoch * len(train_data_loader) + train_idx)
# 				writer.add_scalar('test recall', eval_errors[4], epoch * len(train_data_loader) + train_idx)       
# 			iter_data_time = time.time()
# 		save_path = os.path.join(opt.results_path, opt.name, epoch, f'pred{epoch}.ply')
# 		r = res[0].cpu()
# 		points = samples_tensor[0].transpose(0, 1).cpu()
# 		save_samples_truncated_prob(save_path, points.detach().numpy(), r.detach().numpy())
# 		torch.save(net.state_dict(), os.path.join(opt.checkpoints_path, opt.name, 'net_latest'))
# 		torch.save(net.state_dict(), os.path.join(opt.checkpoints_path, opt.name, f'net_epoch_{epoch}'))
# 		scheduler.step()

if __name__ == '__main__':
	train(opt)




