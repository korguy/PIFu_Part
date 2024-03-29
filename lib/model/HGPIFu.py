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

class HGPIFu(BasePIFuNet):
    '''
    HGPIFuPTF uses stacked hourglass as an image encoder
    + per part training has been added
    '''

    def __init__(self, 
                opt, 
                projection_mode='orthogonal', 
                criteria={'occ': nn.MSELoss()}
                ):
        super(HGPIFu, self).__init__(
            projection_mode=projection_mode,
            criteria=criteria)

        self.name = 'orig_pifu'

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
                            self.opt.load_netF_checkpoint_path, map_location=torch.device(f"cuda")))
        self.netB.load_state_dict(torch.load(
                            self.opt.load_netB_checkpoint_path, map_location=torch.device(f"cuda")))
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

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        give 3d points, obtain 2d projection of these given the camera matrices.
        filter needs to be called beforehand
        then get parts 
        the prediction is stored to self.preds
        args:
            points: [B, 3, N] 3d points in world space
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: [B, 2, 3] image space coordinate transforms
            labels: [B, C, N] ground truth labels (for supervision)
            parts: [B, 26, N] ground truth parts (for supervision)
        '''

        self.intermediate_preds_list = []    

        xyz = self.projection(points, calibs, transforms)
        xy = xyz[:, :2, :]

        # if the point is outside bounding box, return outside.
        in_bb = (xyz >= -1) & (xyz <= 1)
        in_bb = in_bb[:, 0, :] & in_bb[:, 1, :] & in_bb[:, 2, :]
        in_bb = in_bb[:, None, :].detach().float()

        if labels is not None:
            self.lables = in_bb * labels   
        
        sp_feat = self.spatial_enc(xyz, calibs=calibs)

        for i, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), sp_feat]
            point_local_feat = torch.cat(point_local_feat_list, 1)
            pred, _ = self.mlp(point_local_feat)
            pred = in_bb * pred

            self.intermediate_preds_list.append(pred)

        self.preds = self.intermediate_preds_list[-1]

    def get_error(self):
        '''
        Hourglass has its own intermediate supervision scheme
        '''
        error = 0
        for preds in self.intermediate_preds_list:
            error += self.criteria['occ'](preds, self.labels)
        error /= len(self.intermediate_preds_list)
        
        return error

    def get_im_feat(self):
        return self.im_feat_list[-1]


    def forward(self, images, points, calibs, labels, transforms=None):
        self.labels = labels

        self.filter(images)

        self.query(points=points, calibs=calibs, transforms=transforms, labels=labels)

        res = self.get_preds()
        error = self.get_error()

        return res, error
