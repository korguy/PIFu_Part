'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure
import os
from PIL import Image
import cv2
from collections import Counter

from numpy.linalg import inv

colorMap = {
    0 : [0.125, 0., 0.],
    1 : [0.25, 0., 0.],
    2: [0.375, 0., 0.],
    3: [0.5, 0., 0.],
    4: [0.625, 0., 0.],
    5: [0.75, 0., 0.],
    6: [0.875, 0., 0.],
    7: [1., 0., 0.],
    8: [0., 0.125, 0.],
    9: [0., 0.25, 0.],
    10: [0., 0.375, 0.],
    11: [0., 0.5, 0.],
    12: [0., 0.625, 0.],
    13: [0., 0.75, 0.],
    14: [0., 0.875, 0.],
    15: [0., 1.0, 0.],
    16: [0., 0., 0.125],
    17: [0., 0., 0.25],
    18: [0., 0., 0.375],
    19: [0., 0., 0.5],
    20: [0., 0., 0.625],
    21: [0., 0., 0.75],
    22: [0., 0., 0.875],
    23: [0., 0., 1.],
    24: [1., 1., 1.]
}

def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def reconstruction(net, cuda, calib_tensor,
                   resolution, b_min, b_max, thresh=0.5,
                   use_octree=False, num_samples=10000, transform=None):
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    coords, mat = create_grid(resolution, resolution, resolution)
                              #b_min, b_max, transform=transform)

    calib = calib_tensor[0].cpu().numpy()

    calib_inv = inv(calib)
    coords = coords.reshape(3,-1).T
    coords = np.matmul(np.concatenate([coords, np.ones((coords.shape[0],1))], 1), calib_inv.T)[:, :3]
    coords = coords.T.reshape(3,resolution,resolution,resolution)

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, 1, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        
        net.query(samples, calib_tensor)
        pred = net.get_preds()[0][0]
        return pred.detach().cpu().numpy()

    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)

    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, thresh)
        # transform verts into world coordinate system
        trans_mat = np.matmul(calib_inv, mat)
        verts = np.matmul(trans_mat[:3, :3], verts.T) + trans_mat[:3, 3:4]
        verts = verts.T
        # in case mesh has flip transformation
        if np.linalg.det(trans_mat[:3, :3]) < 0.0:
            faces = faces[:,::-1]
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1


def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                continue
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()

def gen_mesh_color(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    img = data['img'].to(device=cuda)
    calib = data['calib'].to(device=cuda)

    net.filter(img)

    try:
        if net.netF is not None:
            img = torch.cat([img, net.nmlF], 0)
        if net.netB is not None:
            img = torch.cat([img, net.nmlB], 0)
    except:
        pass

    save_dir = "/".join(save_path.split("/")[:-1])
    name = save_path.split("/")[-1].split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
                
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_dir, name+".jpg")
    save_img_list = []
    for v in range(img.shape[0]):
        save_img = (np.transpose(img[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    cv2.imwrite(save_img_path, save_img)

    verts, faces, _, _ = reconstruction(net, cuda, calib, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
    
    color = np.zeros(verts.shape)
    interval = 10000
    f = lambda x : colorMap[x]
    for i in range(len(color) // interval):
        left = i * interval
        right = i * interval + interval
        if i == len(color) // interval - 1:
            right = -1
        net.query(verts_tensor[:, :, left:right], calib)
        part = net.get_part()[0].detach().cpu().numpy()
        rgb = list(map(f, part))
#         print(Counter(part))
        color[left:right] = np.array(rgb) * 0.5 + 0.5

    save_obj_mesh_with_color(save_path, verts, faces, color)

def gen_mesh(res, net, cuda, data, save_path, thresh=0.5, use_octree=True, components=False):
    img = data['img'].to(device=cuda)
    calib = data['calib'].to(device=cuda)

    net.filter(img)

    try:
        if net.netF is not None:
            img = torch.cat([img, net.nmlF], 0)
        if net.netB is not None:
            img = torch.cat([img, net.nmlB], 0)
    except:
        pass

    save_dir = "/".join(save_path.split("/")[:-1])
    name = save_path.split("/")[-1].split(".")[0]
    os.makedirs(save_dir, exist_ok=True)
                
    b_min = data['b_min']
    b_max = data['b_max']
    save_img_path = os.path.join(save_dir, name+".jpg")
    save_img_list = []
    for v in range(img.shape[0]):
        save_img = (np.transpose(img[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        save_img_list.append(save_img)
    save_img = np.concatenate(save_img_list, axis=1)
    cv2.imwrite(save_img_path, save_img)

    verts, faces, _, _ = reconstruction(net, cuda, calib, res, b_min, b_max, thresh, use_octree=use_octree, num_samples=50000)
    verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()

    save_obj_mesh(save_path, verts, faces)
