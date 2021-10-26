from torch.utils.data import Dataset

class TrainDataset(Dataset):

	def __init__(self, opt, phase='train'):
		pass

	def get_item(self, index):
		'''
		TODO
		return res according to the format in BaseDataset.py file
		if opt.num_sample_inout, then return samples and labels
		if opt.num_sample_color, then return samples and rgb color values
		'''
		res = {

		}
		return res

	def __getitem__(self, index):
		return self.get_item(index)