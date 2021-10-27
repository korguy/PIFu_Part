import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasePIFuNet import BasePIFuNet
from .MLP import MLP
from .DepthNormalizer import DepthNormalizer
from .HGFilters import HGFilter
from ..net_util import init_net
from ..networks import define_G

class HGPIFuPTF(BasePIFuNet):
	'''
	HGPIFuPTF uses stacked hourglass as an image encoder
	+ per part training has been added
	'''

	def __init__(self, 
				opt, 
				projection_mode='orthogonal', 
				criteria={'occ': nn.MSELoss(),
						  'part': nn.MSELoss()}
				):
		super(HGPIFuPTF, self).__init__(
			projection_mode=projection_mode,
			criteria=criteria)
		self.name = 'pifu_ptf'
		self.parts = None
		in_ch = 3
		try:
			if opt.use_front_normal:
				in_ch += 3
			if opt.use_back_normal:
				in_ch += 3
		except:
			pass

		self.opt = opt
		self.image_filter = HGFilter(opt.num_stack, opt.hg_depth, in_ch, opt.hg_dim,
									opt.norm, opt.hg_down, False)

		self.mlp = MLP(
			filter_channels=self.opt.mlp_dim,
			merge_layer=self.opt.merge_layer,
			res_layers=self.opt.mlp_res_layers,
			norm=self.opt.mlp_norm,
			last_op=nn.Sigmoid())

		self.spatial_enc = DepthNormalizer(opt)

		self.im_feat_list = []
		self.tmpx = None
		self.normx = None
		self.phi = None

		self.intermediate_preds_list = []

		init_net(self)

		self.netF = None
		self.netB = None
		try:
			if opt.use_front_normal:
				self.netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
			if opt.use_back_normal:
				self.netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")
		except:
			pass
		self.nmlF = None
		self.nmlB = None

	def load_FB(self):
		self.netF.load_state_dict(torch.load(
							self.opt.load_netF_checkpoint_path), map_location=torch.device(f"cuda:{self.opt.gpu_id}"))
		self.netB.load_state_dict(torch.load(
							self.opt.load_netF_checkpoint_path), map_location=torch.device(f"cuda:{self.opt.gpu_id}"))
		print("Pix2Pix Network has been loaded.")

	def filter(self, images):
		'''
		apply a fully convolutional network to images.
		the resulting feature will be stored.
		args:
		images: [B, C, H, W]
		'''
		nmls = []
		# if you wish to train jointly, remove detach etc.
		with torch.no_grad():
			if self.netF is not None:
				self.nmlF = self.netF.forward(images).detach()
				nmls.append(self.nmlF)
			if self.netB is not None:
				self.nmlB = self.netB.forward(images).detach()
				nmls.append(self.nmlB)
		if len(nmls) != 0:
			nmls = torch.cat(nmls,1)
			if images.size()[2:] != nmls.size()[2:]:
				nmls = nn.Upsample(size=images.size()[2:], mode='bilinear', align_corners=True)(nmls)
			images = torch.cat([images,nmls],1)


		self.im_feat_list, self.normx = self.image_filter(images)

		if not self.training:
			self.im_feat_list = [self.im_feat_list[-1]]

	def forward(self, images, points, calibs, lables, parts, gamma):
		pass
