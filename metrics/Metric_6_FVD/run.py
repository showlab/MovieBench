import torch
from calculate_fvd import calculate_fvd
import os
from tqdm import tqdm
import numpy as np
import json
# ps: pixel value should be in [0, 1]!
from utils.read_mp4_as_img import video_to_frames_average ,read_frames


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
        "--Image_Format",
        type=str,
        required=True,
        help="jpg|png",
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

    
    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]
    Test_Set = ["1004_Juno"]
    
    gt_video_path = args.GT_video_path
    generated_video_path = args.Pre_path


    result = {}
    result_list = {}
    device = torch.device("cuda")
    # device = torch.device("cpu")

    for index,movie in enumerate(tqdm(Test_Set)):
            
        image_root = '{}/{}'.format(gt_video_path,movie)
        file_list = os.listdir(image_root)

        images_list = []
        images_gt_list = []
        for idx,name in enumerate(tqdm(file_list)):
            gt_path = os.path.join(image_root,name)
            pred_path = os.path.join("{}/{}".format(generated_video_path,movie),name.replace("avi","mp4"))

            images_gt = video_to_frames_average(gt_path,(args.Resolution,args.Resolution),args.Frame_Number)
            images = video_to_frames_average(pred_path,(args.Resolution,args.Resolution),args.Frame_Number)

            max_len = min(len(images_gt),len(images))
            images_gt = images_gt[:max_len-1]
            images = images[:max_len-1]

            images = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images]
            images_gt = [torch.tensor(np.array(img).transpose(2, 0, 1)) for img in images_gt]


            videos1 = torch.stack(images)
            videos2 = torch.stack(images_gt)

            images_list.append(torch.stack(images))
            images_gt_list.append(torch.stack(images_gt))


            

        videos1 = torch.stack(images_list).to(torch.float32) / 255
        videos2 = torch.stack(images_gt_list).to(torch.float32) / 255
        
        result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
        # resu = json.dumps(result, indent=4)
        value = result["fvd"]["value"]
        average_value = sum(value.values()) / len(value)
        result_list[movie] = average_value
        print(result_list)

    average_value = sum(result_list.values()) / len(result_list)
    print("Average FVD:", average_value)

