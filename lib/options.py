import argparse
import os 


class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):

		parser.add_argument('--name', type=str, default='ptf_pifu')

		g_data = parser.add_argument_group('Data')
		g_data.add_argument('--dataroot', type=str, default='./training_data')
		g_data.add_argument('--loadSizeSmall', type=int, default=512)
		g_data.add_argument('--loadSizeBig', type=int, default=1024)

		g_train = parser.add_argument_group('Training')
		g_train.add_argument('--gpu_id', type=int, default=0)
		g_train.add_argument('--batch_size', type=int, default=4)
		g_train.add_argument('--learning_rate', type=float, default=1e-3)
		g_train.add_argument('--num_iter', type=int, default=50000)
		g_train.add_argument('--freq_plot', type=int, default=100)
		g_train.add_argument('--freq_mesh', type=int, default=10000)
		g_train.add_argument('--freq_eval', type=int, default=10000)
		g_train.add_argument('--freq_save_ply', type=int, default=10000)
		g_train.add_argument('--resume_epoch', type=int, default=-1)
		g_train.add_argument('--continue_train', action='store_true')
		g_train.add_argument('--serial_batches', action='store_true',
								help='if true, takes images in order to make batches, otherwise takes them randomly')
		g_train.add_argument('--num_threads', default=1, type=int, help='# sthreads for loading data')
		g_train.add_argument('--pin_memory', action='store_true', help='pin_memory')

		g_train.add_argument('--results_path', type=str, default="./runs")
		g_train.add_argument('--num_parts', type=int, default=26)

		g_train.add_argument('--bg_path', type=str, default='./val2017')


		g_test = parser.add_argument_group('Testing')
		g_test.add_argument('--resolution', type=int, default=512)

		g_sample = parser.add_argument_group('Sampling')
		g_sample.add_argument('--num_sample_inout', type=int, default=8000)
		g_sample.add_argument('--num_sample_color', type=int, default=0)	
		g_sample.add_argument('--sigma', type=float, default=1.0, help='sigma for sampling')
		g_sample.add_argument('--sigma_surface', type=float, default=1.0, help='sigma for sampling')

		g_sample.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')

		g_model = parser.add_argument_group('Model')
		g_model.add_argument('--norm', type=str, default='batch')
		g_model.add_argument('--netG', type=str, default='hgpifu')

		g_model.add_argument('--num_stack', type=int, default=4)
		g_model.add_argument('--hg_depth', type=int, default=2)
		g_model.add_argument('--hg_down', type=str, default='ave_pool')
		g_model.add_argument('--hg_dim', type=int, default=256)

		g_model.add_argument('--mlp_norm', type=str, default='batch')
		g_model.add_argument('--mlp_dim', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
			help='# of dimensions of mlp. no need to put the first channel')
		g_model.add_argument('--mlp_dim_color', nargs='+', default=[1024, 512, 256, 128, 3], type=int,
			help='# of dimensions of mlp. no need to put the first channel')
		g_model.add_argument('--mlp_res_layers', nargs='+', default=[2,3,4], type=int,
			help='leyers that has skip connection. use 0 for no residual pass')
		g_model.add_argument('--merge_layer', type=int, default=2)

		parser.add_argument('--random_body_chop', action='store_true', help='if random flip')
		parser.add_argument('--random_flip', action='store_true', help='if random flip')
		parser.add_argument('--random_trans', action='store_true', help='if random flip')
		parser.add_argument('--random_scale', action='store_true', help='if random flip')
		parser.add_argument('--random_rotate', action='store_true', help='if random flip')
		parser.add_argument('--random_bg', action='store_true', help='using random background')

		parser.add_argument('--schedule', type=int, nargs='+', default=[10, 15],
			help='Decrease learning rate at these epochs.')
		parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
		parser.add_argument('--lambda_nml', type=float, default=0.0, help='weight of normal loss')
		parser.add_argument('--lambda_cmp_l1', type=float, default=0.0, help='weight of normal loss')
		parser.add_argument('--occ_loss_type', type=str, default='bce', help='bce | brock_bce | mse')
		parser.add_argument('--clr_loss_type', type=str, default='mse', help='mse | l1')
		parser.add_argument('--occ_gamma', type=float, default=None, help='weighting term')

		# for eval
		parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
		parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
		parser.add_argument('--gen_test_mesh', action='store_true', help='generate test mesh')
		parser.add_argument('--gen_train_mesh', action='store_true', help='generate train mesh')
		parser.add_argument('--all_mesh', action='store_true', help='generate meshs from all hourglass output')
		parser.add_argument('--num_gen_mesh_test', type=int, default=4,
			help='how many meshes to generate during testing')
		parser.add_argument('--load_checkpoints_path', type=str, default="./checkpoints")
		parser.add_argument('--load_netF_checkpoint_path', type=str, default="./checkpoints/pix2pix/netF", help='path to save checkpoints')
		parser.add_argument('--load_netB_checkpoint_path', type=str, default="./checkpoints/pix2pix/netB", help='path to save checkpoints')
		parser.add_argument('--use_aio_normal', action='store_true')
		parser.add_argument('--use_front_normal', action='store_true', default=True)
		parser.add_argument('--use_back_normal', action='store_true', default=True)
		parser.add_argument('--no_intermediate_loss', action='store_true')

		# aug
		group_aug = parser.add_argument_group('aug')
		group_aug.add_argument('--aug_alstd', type=float, default=0.0, help='augmentation pca lighting alpha std')
		group_aug.add_argument('--aug_bri', type=float, default=0.2, help='augmentation brightness')
		group_aug.add_argument('--aug_con', type=float, default=0.2, help='augmentation contrast')
		group_aug.add_argument('--aug_sat', type=float, default=0.05, help='augmentation saturation')
		group_aug.add_argument('--aug_hue', type=float, default=0.05, help='augmentation hue')
		group_aug.add_argument('--aug_gry', type=float, default=0.1, help='augmentation gray scale')
		group_aug.add_argument('--aug_blur', type=float, default=0.0, help='augmentation blur')
		# special tasks
		self.initialized = True
		return parser

	def gather_options(self, args=None):
		if not self.initialized:
			parser = argparse.ArgumentParser(
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		parser = self.initialize(parser)
		self.parser = parser

		if args is None:
			return self.parser.parse_args()
		else:
			return self.parser.parse_args(args)

	def print_options(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
		if v != default:
			comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
			message += '----------------- End -------------------'
			print(message)

	def parse(self, args=None):
		opt = self.gather_options(args)
        
		if len(opt.mlp_res_layers) == 1 and opt.mlp_res_layers[0] < 1:
			opt.mlp_res_layers = []

		return opt		
