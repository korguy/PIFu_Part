import numpy as np
import os 
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
import glob 
import random

def augment_bg(base_path, subject_name, bg_img_list, imgsize):
    origin = os.path.join(base_path, "RENDER_ORIGIN", subject_name)
    new_path = os.path.join(base_path, "RENDER", subject_name)
    os.makedirs(origin, exist_ok=True)
    try:
        os.rename(new_path, origin)
    except Exception as e:
        print(e)
        return
    render_img_list = glob.glob(os.path.join(origin, '*.jpg'))
    render_img_list.sort()
    mask_img_list = glob.glob(os.path.join(base_path, "MASK", subject_name, "*.png"))
    mask_img_list.sort()
    os.makedirs(new_path, exist_ok=True)
    np.random.seed(random.randrange(1997))
    for i in range(len(render_img_list)):
        render_img = np.asarray(Image.open(render_img_list[i])).astype(np.float32)
        mask_img = np.asarray(Image.open(mask_img_list[i])).reshape([imgsize, imgsize, 1]).astype(np.float32)
        bg_img = np.asarray(Image.open(bg_img_list[np.random.randint(len(bg_img_list))]).resize((imgsize, imgsize)))
        while len(bg_img.shape) != 3:
            bg_img = np.asarray(Image.open(bg_img_list[np.random.randint(len(bg_img_list))]).resize((imgsize, imgsize)))
        new_img = (render_img + (1 - mask_img / 255.) * bg_img).astype(np.uint8)
        Image.fromarray(new_img).save(os.path.join(new_path, "%d_0_00.jpg"%i))

def main(args):
    subjects = glob.glob(os.path.join(args.input, "RENDER", "*"))
    bg_img_list = glob.glob(os.path.join(args.bg_path, "*.jpg"))
    bg_img_list.sort()   
    print("Augmenting with %d Images..." % len(bg_img_list))
    for subject in tqdm(subjects, total=len(subjects)):
        try:
            name = subject.split("/")[-1]
            augment_bg(args.input, name, bg_img_list, args.size)
        except:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./training_data")
    parser.add_argument('-b', '--bg_path', type=str, default='./val2017')
    parser.add_argument('-s', '--size', type=int, default=1024)
    args = parser.parse_args()
    
    main(args)