import torch 
import numpy as np


def compute_acc(pred, gt, thresh=0.5):
	with torch.no_grad():
		vol_pred = pred > thresh
		vol_gt = gt > thresh

		union = vol_pred | vol_gt
		inter = vol_pred & vol_gt

		true_pos = inter.sum().float()
		union = union.sum().float()
		if union == 0:
			union = 1
		vol_pred = vol_pred.sum().float()
		if vol_pred == 0:
			vol_pred = 1
		vol_gt = vol_gt.sum().float()
		if vol_gt == 0:
			vol_gt = 1
		return true_pos / union, true_pos / vol_pred, true_pos / vol_gt