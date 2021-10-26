from torch.utils.data import Dataset


class EvalDataset(Dataset):

	def __init__(self, opt, root=None):
		pass

	def get_item(self, index):
		'''
		TODO
		res(dict) : returned data
		res should have name, mesh_path, img, calib, extrinsic, mask
		'''
		pass

	def __getitem__(self, index):
		return self.get_item(index)