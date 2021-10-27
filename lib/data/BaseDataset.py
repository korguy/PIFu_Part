from torch.utils.data import Dataset
import random 

class BaseDataset(Dataset):
	'''
	This is the Base Datasets.
	Just for inference of get_item function's return type
	'''

	@staticmethod
	def modify_commandline_options(parser, is_train):
		return parser

	def __init__(self, opt, phase='train'):
		self.opt = opt
		self.is_train = self.phase == 'train'
		self.projection_mode = 'orthogonal'

	def __len__(self):
		return 0

	def get_item(self, index):
		try:
			res = {
				'name' : None, # name of the RP subject
				'b_min' : None, # Bounding box (x_min, y_min, z_min) of target space
				'b_max' : None, # Bounding box (x_max, y_max, z_max) of target space

				'samples' : None, # [3, N] samples 
				'labels' : None, # [1, N] labels
				'parts' : None, # [num_parts, N] parts 

				'img' : None, # [C, H, W] input images 512x512 for shape, 1024x1024 for color
				'calib' : None, # [4, 4] calibration matrix
				'extrinsic' : None, # [4, 4] extrinsic matrix
				'mask' : None # [1, H, W] segmentation masks
			}

			return res
		except:
			print(f"Requested index {index} has missing files. Using a random sample instead.")
			return self.get_item(index=random.randint(0, self.__len__() - 1))

	def __getitem__(self, index):
		return self.get_item(index)


folder
 -folder
  - abc.obj
  - abc.pkl 
  ...
  ...
  ...
-folder
