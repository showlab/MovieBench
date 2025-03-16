import torch
import numpy as np
import os
from utils.read_mp4_as_img import video_to_frames,read_frames,video_to_frames_average
from torchmetrics.image.inception import InceptionScore
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


if __name__ == '__main__':
    args = parse_args()
    _ = torch.manual_seed(123)

    inception = InceptionScore()

    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]

    sum_score = 0
    number = 0

    for movie in tqdm(Test_Set):
        
        image_root = '{}/{}'.format(args.Pre_path,movie)
        file_list = os.listdir(image_root)

        for name in file_list:

            if args.Format == "Image":
                image_path = os.path.join(image_root,name)
                images = read_frames(image_path,[args.Resolution,args.Resolution])

                images = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images]
                images = torch.stack(images).to(torch.uint8)
            else:
                image_path = os.path.join(image_root,name)
                images_list = video_to_frames_average(image_path,(args.Resolution,args.Resolution),args.Frame_Number)
                images = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images_list]
                images = torch.stack(images).to(torch.uint8)

            inception.update(images)
            is_mean, is_std = inception.compute()
    print("mean:",is_mean,"std:",is_std)
