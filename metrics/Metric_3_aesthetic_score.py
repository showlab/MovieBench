import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from utils.read_mp4_as_img import video_to_frames,read_frames,video_to_frames_average
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



def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m



if __name__ == '__main__':
    device = 'cuda'
    args = parse_args()
    dtype = torch.float32
    
    clip_path = './weights/clip-vit-large-patch14'
    
    pre_precessor = CLIPImageProcessor.from_pretrained(clip_path)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(clip_path)
    model = get_aesthetic_model()
    
    clip_model = clip_model.to(device, dtype)
    model = model.to(device, dtype)
    
    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]
    sum_score = 0
    number = 0

    for movie in tqdm(Test_Set):

        image_root = '{}/{}'.format(args.Pre_path,movie)
        file_list = os.listdir(image_root)

        sum_score = 0
        number = 0
        for name in tqdm(file_list):
            if args.Format == "Image":
                image_path = os.path.join(image_root,name)
                images = read_frames(image_path,[args.Resolution,args.Resolution])
            else:
                image_path = os.path.join(image_root,name)
                images = video_to_frames_average(image_path,(args.Resolution,args.Resolution),args.Frame_Number)

            tensor = pre_precessor.preprocess(images, return_tensors='pt').pixel_values.to(device, dtype)
        
            tmp = clip_model(tensor).image_embeds
            tmp = model(tmp)
            
            score = tmp.mean()
            number += 1
            sum_score += score.item()
    print(sum_score/number)
