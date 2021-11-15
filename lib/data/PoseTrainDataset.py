from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import json
from math import sqrt
import datetime

log = logging.getLogger('trimesh')
log.setLevel(40)

def get_part(file, vertices, points, body_parts):
	def get_dist(pt1, pt2):
		return sqrt((pt1[0]-pt2[0])**2 +(pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
	part = []
	for point in points:
		_min = float('inf')
		_idx = 0
		for idx, vertice in enumerate(vertices[::5]):
			dist = get_dist(point, vertice)
			if _min > dist:
				_min = dist
				_idx = idx
		tmp = [0 for i in range(20)]
		tmp[ body_parts.index(file[str(_idx)]) ] = 1 # one-hot vector making
		part.append(tmp)
	part = np.array(part)
	return part

def load_trimesh(root_dir):
	folders = os.listdir(root_dir)
	meshs = {}
	for i, f in enumerate(folders):
		sub_name = f
		meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_posed.obj' % sub_name), process=False, maintain_order=True,
									  skip_uv=True)
		#### mesh = trimesh.load("alvin_t_posed.obj",process=False, maintain_order=True, skip_uv=True)

	return meshs

def save_samples_truncated_part(fname, points, part):
	'''
	points: [N, 3] points sampled from mesh
	part: [N, 20] one hot vector representation
	'''
	part_idx = np.argmax(part, axis=1)
	r = ((255/20)*part_idx).reshape((-1, 1))
	g = (255//(part_idx+1)).reshape((-1, 1))
	b = (255/20*(20-part_idx)).reshape((-1, 1))
	
	to_save = np.concatenate([points, r,g, b], axis=-1)
	return np.savetxt(fname,
				  to_save,
				  fmt='%.6f %.6f %.6f %d %d %d',
				  comments='',
				  header=(
					  'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
					  points.shape[0])
				  )
	
def save_samples_truncted_prob(fname, points, prob):
	'''
	Save the visualization of sampling to a ply file.
	Red points represent positive predictions.
	Green points represent negative predictions.
	:param fname: File name to save
	:param points: [N, 3] array of points
	:param prob: [N, 1] array of predictions in the range [0~1]
	:return:
	'''
	r = (prob > 0.5).reshape([-1, 1]) * 255
	g = (prob < 0.5).reshape([-1, 1]) * 255
	b = np.zeros(r.shape)

	to_save = np.concatenate([points, r, g, b], axis=-1)
	return np.savetxt(fname,
					  to_save,
					  fmt='%.6f %.6f %.6f %d %d %d',
					  comments='',
					  header=(
						  'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
						  points.shape[0])
					  )

class PoseTrainDataset(Dataset):

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def __init__(self, opt, phase='train'):
		self.opt = opt
		self.projection_mode = 'orthogonal'

		# Path setup
		self.root = self.opt.dataroot
		self.RENDER = os.path.join(self.root, 'RENDER')
		self.PART = os.path.join(self.root, 'PART')
		self.MASK = os.path.join(self.root, 'MASK')
		self.PARAM = os.path.join(self.root, 'PARAM')
		self.UV_MASK = os.path.join(self.root, 'UV_MASK')
		self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
		self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
		self.UV_POS = os.path.join(self.root, 'UV_POS')
		self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

		self.BG = self.opt.bg_path
		self.bg_img_list = []
		if self.opt.random_bg:
			self.bg_img_list = [os.path.join(self.BG, x) for x in os.listdir(self.BG)]
			self.bg_img_list.sort()

		self.B_MIN = np.array([-128, -28, -128]) / 128
		self.B_MAX = np.array([128, 228, 128]) / 128
		self.num_views = 1
		self.is_train = (phase == 'train')
		self.load_size = self.opt.loadSizeSmall
	  

		self.num_sample_inout = self.opt.num_sample_inout
		self.num_sample_color = self.opt.num_sample_color

		self.yaw_list = list(range(0,360,1))
		self.pitch_list = [0]
		self.subjects = self.get_subjects()

		# PIL to tensor
		self.to_tensor = transforms.Compose([
			transforms.Resize(self.load_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])

		# augmentation
		self.aug_trans = transforms.Compose([
			transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
								   hue=opt.aug_hue)
		])

		self.mesh_dic = load_trimesh(self.OBJ)

	def get_subjects(self):
		all_subjects = os.listdir(self.RENDER)
		var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
		if len(var_subjects) == 0:
			return all_subjects

		if self.is_train:
			return sorted(list(set(all_subjects) - set(var_subjects)))
		else:
			return sorted(list(var_subjects))

	def __len__(self):
		return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

	def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
		'''
		Return the render data
		:param subject: subject name
		:param num_views: how many views to return
		:param view_id: the first view_id. If None, select a random one.
		:return:
			'img': [num_views, C, W, H] images
			'calib': [num_views, 4, 4] calibration matrix
			'extrinsic': [num_views, 4, 4] extrinsic matrix
			'mask': [num_views, 1, W, H] masks
		'''
		pitch = self.pitch_list[pid]

		# The ids are an even distribution of num_views around view_id
		view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
					for offset in range(num_views)]
		if random_sample:
			view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

		calib_list = []
		render_list = []
		mask_list = []
		extrinsic_list = []
		
		vid = 0

		param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
		render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
		mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))

		# loading calibration data
		param = np.load(param_path, allow_pickle=True)
		# pixel unit / world unit
		ortho_ratio = param.item().get('ortho_ratio')
		# world unit / model unit
		scale = param.item().get('scale')
		# camera center world coordinate
		center = param.item().get('center')
		# model rotation
		R = param.item().get('R')

		translate = -np.matmul(R, center).reshape(3, 1)
		extrinsic = np.concatenate([R, translate], axis=1)
		extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
		# Match camera space to image pixel space
		scale_intrinsic = np.identity(4)
		scale_intrinsic[0, 0] = scale / ortho_ratio
		scale_intrinsic[1, 1] = -scale / ortho_ratio
		scale_intrinsic[2, 2] = scale / ortho_ratio
		# Match image pixel space to image uv space
		uv_intrinsic = np.identity(4)
		uv_intrinsic[0, 0] = 1.0 / float(self.load_size // 2)
		uv_intrinsic[1, 1] = 1.0 / float(self.load_size // 2)
		uv_intrinsic[2, 2] = 1.0 / float(self.load_size // 2)
		# Transform under image pixel space
		trans_intrinsic = np.identity(4)

		mask = Image.open(mask_path).convert('L')
		render = Image.open(render_path).convert('RGB')

		if self.is_train:
			# Pad images
			pad_size = int(0.1 * self.load_size)
			render = ImageOps.expand(render, pad_size, fill=0)
			mask = ImageOps.expand(mask, pad_size, fill=0)

			w, h = render.size
			th, tw = self.load_size, self.load_size

			# random flip
			if self.opt.random_flip and np.random.rand() > 0.5:
				scale_intrinsic[0, 0] *= -1
				render = transforms.RandomHorizontalFlip(p=1.0)(render)
				mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

			# random scale
			if self.opt.random_scale:
				rand_scale = random.uniform(0.9, 1.1)
				w = int(rand_scale * w)
				h = int(rand_scale * h)
				render = render.resize((w, h), Image.BILINEAR)
				mask = mask.resize((w, h), Image.NEAREST)
				scale_intrinsic *= rand_scale
				scale_intrinsic[3, 3] = 1

			# random translate in the pixel space
			if self.opt.random_trans:
				dx = random.randint(-int(round((w - tw) / 10.)),
									int(round((w - tw) / 10.)))
				dy = random.randint(-int(round((h - th) / 10.)),
									int(round((h - th) / 10.)))
			else:
				dx = 0
				dy = 0

			trans_intrinsic[0, 3] = -dx / float(self.load_size // 2)
			trans_intrinsic[1, 3] = -dy / float(self.load_size // 2)

			x1 = int(round((w - tw) / 2.)) + dx
			y1 = int(round((h - th) / 2.)) + dy

			render = render.crop((x1, y1, x1 + tw, y1 + th))
			mask = mask.crop((x1, y1, x1 + tw, y1 + th))

			render = self.aug_trans(render)

			# random blur
			if self.opt.aug_blur > 0.00001:
				blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
				render = render.filter(blur)

		intrinsic = np.matmul(trans_intrinsic, np.matmul(uv_intrinsic, scale_intrinsic))
		calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
		extrinsic = torch.Tensor(extrinsic).float()

		mask = transforms.Resize(self.load_size)(mask)
		mask = transforms.ToTensor()(mask).float()
		mask_list.append(mask)

		render = self.to_tensor(render)
		render = mask.expand_as(render) * render

		render_list.append(render)
		calib_list.append(calib)
		extrinsic_list.append(extrinsic)

		if self.opt.random_bg: # background에도 augmentation 추가하기
			bg_path = self.bg_img_list[np.random.randint(len(self.bg_img_list))]
			bg = Image.open(bg_path).convert('RGB').resize((self.load_size, self.load_size), Image.NEAREST)
			
			bg = self.to_tensor(bg)
			
			render = (1-mask).expand_as(render) * bg + render

		return {
			'img': render_list[0],
			'calib': calib_list[0],
			'extrinsic': extrinsic_list[0],
			'mask': mask_list[0]
		}

	def select_sampling_method(self, subject):
		if not self.is_train:
			random.seed(1997)
			np.random.seed(1997)
			torch.manual_seed(1997)
		mesh = self.mesh_dic[subject]
		surface_points, surface_points_face_indices = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
		sample_points = surface_points + np.random.normal(scale=(self.opt.sigma / 128.), size=surface_points.shape)
		
		body_parts = [ 'head', 'neck','spine', 'hip', 
					   'shoulder_l', 'upperarm_l', 'lowerarm_l', 'hand_l', 'finger_l',
					   'shoulder_r', 'upperarm_r', 'lowerarm_r', 'hand_r', 'finger_r',  
					   'upperleg_l', 'lowerleg_l', 'foot_l', 
					   'upperleg_r', 'lowerleg_r', 'foot_r' ] 
		
		# one-hot vectors of the sampled points
		surface_points_body_parts = []
		
		with open(os.path.join(self.PART, subject, "%s_part.json" % subject.split('_')[0])) as f: 
			json_data = json.load(f)
		ref = max([int(x) for x in json_data.keys()])
		surface_points_faces = mesh.faces[surface_points_face_indices]
		surface_points_vertices_indices = []
		
		for single_face in surface_points_faces:
			surface_points_vertices_indices.append(min(single_face)) # take the first vertex of the face as a representative
		
   
		for idx_num in surface_points_vertices_indices:
		try:
			idx = str(idx_num)
			temp = [0 for i in range(20)]
			temp[ body_parts.index(json_data[idx]) ] = 1 # one-hot vector making
			surface_points_body_parts.append(temp)
		except:
			print(subject)
			print(ref, idx)
			
		# add random points within image space
		length = self.B_MAX - self.B_MIN
		### New
		random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
		sample_points = np.concatenate([sample_points, random_points], 0)
		random_parts = get_part(json_data, mesh.vertices, random_points, body_parts)
		surface_points_body_parts = np.concatenate([surface_points_body_parts, random_parts], 0)
		###
#         for i in range(0, self.num_sample_inout // 4):
#             surface_points_body_parts.append([0 for i in range(20)]) # append zero vectors [0, 0, 0, ... , 0] 
			
		s = np.arange(sample_points.shape[0])
		np.random.shuffle(s)
		sample_points = sample_points[s]
		surface_points_body_parts = np.array(surface_points_body_parts)[s]
		#np.random.shuffle(sample_points)
				
		inside = mesh.contains(sample_points)
		inside_points = sample_points[inside]
		inside_parts = surface_points_body_parts[inside]
		outside_points = sample_points[np.logical_not(inside)]
		outside_parts = surface_points_body_parts[np.logical_not(inside)]

		nin = inside_points.shape[0]
		inside_points = inside_points[
						:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
		inside_parts = inside_parts[
						:self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_parts
		outside_points = outside_points[
						 :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
																							   :(self.num_sample_inout - nin)]
		outside_parts = outside_parts[
						 :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_parts[
																							   :(self.num_sample_inout - nin)]

		samples = np.concatenate([inside_points, outside_points], 0).T
		labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)
		parts = np.concatenate([inside_parts, outside_parts], 0).T

#         save_samples_truncted_prob('./out.ply', samples.T, labels.T)
#         exit()
#         save_samples_truncated_part('./part.ply', samples.T, parts.T)
#         exit()

		samples = torch.Tensor(samples).float()
		labels = torch.Tensor(labels).float()
		parts = torch.Tensor(parts).float()
		
		del mesh

		return {
			'samples': samples,
			'labels': labels,
			'parts': parts
		}


	def get_color_sampling(self, subject, yid, pid=0):
		yaw = self.yaw_list[yid]
		pitch = self.pitch_list[pid]
		uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
		uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
		uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
		uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

		# Segmentation mask for the uv render.
		# [H, W] bool
		uv_mask = cv2.imread(uv_mask_path)
		uv_mask = uv_mask[:, :, 0] != 0
		# UV render. each pixel is the color of the point.
		# [H, W, 3] 0 ~ 1 float
		uv_render = cv2.imread(uv_render_path)
		uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

		# Normal render. each pixel is the surface normal of the point.
		# [H, W, 3] -1 ~ 1 float
		uv_normal = cv2.imread(uv_normal_path)
		uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
		uv_normal = 2.0 * uv_normal - 1.0
		# Position render. each pixel is the xyz coordinates of the point
		uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

		### In these few lines we flattern the masks, positions, and normals
		uv_mask = uv_mask.reshape((-1))
		uv_pos = uv_pos.reshape((-1, 3))
		uv_render = uv_render.reshape((-1, 3))
		uv_normal = uv_normal.reshape((-1, 3))

		surface_points = uv_pos[uv_mask]
		surface_colors = uv_render[uv_mask]
		surface_normal = uv_normal[uv_mask]

		if self.num_sample_color:
			sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
			surface_points = surface_points[sample_list].T
			surface_colors = surface_colors[sample_list].T
			surface_normal = surface_normal[sample_list].T

		# Samples are around the true surface with an offset
		normal = torch.Tensor(surface_normal).float()
		samples = torch.Tensor(surface_points).float() \
				  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal

		# Normalized to [-1, 1]
		rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

		return {
			'color_samples': samples,
			'rgbs': rgbs_color
		}

	def get_item(self, index):
		# In case of a missing file or IO error, switch to a random sample instead
		# try:
		sid = index % len(self.subjects)
		tmp = index // len(self.subjects)
		yid = tmp % len(self.yaw_list)
		pid = tmp // len(self.yaw_list)

		# name of the subject 'rp_xxxx_xxx'
		subject = self.subjects[sid]
		res = {
			'name': subject,
			'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
			'sid': sid,
			'yid': yid,
			'pid': pid,
			'b_min': self.B_MIN,
			'b_max': self.B_MAX,
		}
		render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid)
		res.update(render_data)

		if self.opt.num_sample_inout:
			sample_data = self.select_sampling_method(subject)
			res.update(sample_data)
		
		# img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
		# rot = render_data['calib'][0,:3, :3]
		# trans = render_data['calib'][0,:3, 3:4]
		# pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
		# pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
		# for p in pts:
		#     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
		# cv2.imshow('test', img)
		# cv2.waitKey(1)

		if self.num_sample_color:
			color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
			res.update(color_data)
		return res
		# except Exception as e:
		#     print(e)
		#     return self.get_item(index=random.randint(0, self.__len__() - 1))

	def __getitem__(self, index):
		return self.get_item(index)
