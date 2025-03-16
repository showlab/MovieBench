from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import CLIPImageProcessor
import os
from utils.read_mp4_as_img import video_to_frames,read_frames, video_to_frames_average
import json
from tqdm import tqdm
# change this model path
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
        "--GT_json_path",
        type=str,
        required=True,
        help="The path for the groundtruth",
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


    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    pre_precessor = CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')
    sum_score = 0
    number = 0
    
    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]

    for movie in Test_Set:

        root = '{}/{}'.format(args.GT_json_path,movie)
        file_list = os.listdir(root)

        for name in tqdm(file_list):
            json_path = os.path.join(root,name)
            
            if args.Format == "Image":
                image_path = os.path.join('{}/{}'.format(args.Pre_path,movie),name.replace("json","jpg"))

                images = read_frames(image_path,[args.Resolution,args.Resolution])

                tensor = pre_precessor.preprocess(images, return_tensors='pt', do_normalize=False).pixel_values
            
            else:
                image_path = os.path.join('{}/{}'.format(args.Pre_path,movie),name.replace("json","mp4"))
                images_list = video_to_frames_average(image_path,(args.Resolution,args.Resolution),args.Frame_Number)
                tensor = pre_precessor.preprocess(images_list, return_tensors='pt', do_normalize=False).pixel_values

            with open(json_path, 'r') as file:
                ann = json.load(file)

            Character_str = ann["Plot"]
            text = [Character_str] * tensor.shape[0]

            score = metric(tensor, text)
            number += 1
            sum_score += score.item()

    print(sum_score/number)