'''
current data structure:
	subjects
		|--tex
		|--pose.obj
		|--t-pose.obj
'''
import shutil
import os
import argparse


def reorganize(base, name):
	part_file = os.path.join(base, name, f"{name}_part.json")
	tex_folder = os.path.join(base, name, "tex")
	t_pose = os.path.join(base, name, f"{name}_t_posed.obj")
	poses = [x for x in os.listdir(os.path.join(base, name)) if "posed.obj" in x]
	for pose in poses:
		pose_name = pose[:-10]
		new_path = os.path.join(base, pose_name)
		os.makedirs(new_path, exist_ok=True)
		shutil.move(os.path.join(base, name, pose), os.path.join(new_path, name))
		shutil.copyfile(part_file, os.path.join(new_path, f"{name}_part.json"))
		shutil.copyfile(t_pose, os.path.join(new_path, f"{name}_t_pose.obj"))
		shutil.copytree(tex_folder, os.path.join(new_path, "tex"))
	os.rename(os.path.join(base, name), os.path.join(base, f"{name}_base"))


def main(args):
	path = args.input
	subjects = os.listdir(path)
	subjects = [x for x in subjects if "." not in x]
	for subject in subjects:
		reorganize(path, subject)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", type=str, default="../data")
	args = parser.parse_args()
	main(args)
