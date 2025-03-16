# 有问题见
# https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html

import torch
import numpy as np
import os
from utils.read_mp4_as_img import video_to_frames ,read_frames, video_to_frames_average
from torchmetrics.image.fid import FrechetInceptionDistance
import cv2
from tqdm import tqdm
import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description="MovieBench")

    parser.add_argument(
        "--Pre_path",
        type=str,
        required=True,
        help="The path for predicted output",
    )
    parser.add_argument(
        "--GT_video_path",
        type=str,
        required=True,
        help="The path for video GT",
    )
    parser.add_argument(
        "--Format",
        type=str,
        required=True,
        help="Image|Video",
    )
    parser.add_argument(
        "--Resolution",
        type=int,
        required=False,
        help="256",
    )
    parser.add_argument(
        "--Frame_Number",
        type=int,
        required=False,
        default=25,
        help="256",
    )

    args = parser.parse_args()

    return args

def get_random_frame_from_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    random_frame_index = int(total_frames/2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index)

    ret, frame = cap.read()

    if ret:
        cap.release()
        return [frame]
    else:
        print(f"Error reading frame at index {random_frame_index}")
        cap.release()
        return None

if __name__ == '__main__':
    device = 'cuda'
    args = parse_args()
        
    _ = torch.manual_seed(123)

    # 64, 192, 768, 2048
    fid = FrechetInceptionDistance(feature=192, input_img_size=(3, args.Resolution, args.Resolution))     # 参数可以看上面网站

    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]

    sum_score = 0
    number = 0

    for index,movie in enumerate(tqdm(Test_Set)):
            
        video_root = '{}/{}'.format(args.GT_video_path,movie)
        file_list = os.listdir(video_root)

        images_list = []
        images_gt_list = []
        for idx,name in enumerate(tqdm(file_list)):
            video_path = os.path.join(video_root,name)
            GT_path = os.path.join("{}/{}".format(args.GT_video_path,movie),name)

            if args.Format == "Image":
                image_path = os.path.join("{}/{}".format(args.Pre_path,movie),name.replace(".avi",".jpg"))
                images_gt = get_random_frame_from_video(GT_path)
                images = read_frames(image_path)
            else:
                image_path = os.path.join("{}/{}".format(args.Pre_path,movie),name.replace(".avi",".mp4"))

                images_gt = video_to_frames_average(GT_path,(args.Resolution,args.Resolution),args.Frame_Number)
                images = video_to_frames_average(image_path,(args.Resolution,args.Resolution),args.Frame_Number)

                max_len = min(len(images_gt),len(images))
                images_gt = images_gt[:max_len-1]
                images = images[:max_len-1]

                images = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images]
                images_gt = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images_gt]

            images_list += images
            images_gt_list += images_gt
        
        images = torch.stack(images_list).to(torch.uint8)
        images_gt = torch.stack(images_gt_list).to(torch.uint8)

        fid.update(images, real=False)
        fid.update(images_gt, real=True)

    score = fid.compute()
    print("FID:",score)