import time
import os
import json
import imageio
import requests
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from utils import encode_image
from transformers import AutoModel, AutoTokenizer
import openai
from openai import OpenAI
import torch 
from PIL import Image
import vertexai

import IPython.display
from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
   

class MovieSeq_gemini_1_5_pro:
    def __init__(self,
                    model="gemini-1.5-pro-001", PROJECT_ID = "1009298643129" ,api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail


        # PROJECT_ID = "1009298643129"  # @param {type:"string"}
        # LOCATION = "us-central1"  # @param {type:"string"}

        # import vertexai
        vertexai.init(project=PROJECT_ID)

        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A sequence of photos of characters along with their names.
        2. Keyframes from a video clip.
        3. The corresponding dialogues from the video clip, each associated with a speaker ID.
        4. Manually annotated Movie Visual Description, including the narration for the visual elements in movies.

        Your task is to analyze and associate these inputs, understand the context of the video, and respond to the user's needs accordingly.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = GenerativeModel(model)

    def get_response(self, char_bank, frame_list, diag_list, Need_AD,
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [self.system_text]
        

        # char_list = []
        for char_name, char_url in char_bank.items():
            images = Part.from_uri(char_url, mime_type="image/png")
            messages.append(f"This is the photo of {char_name}.")
            messages.append(images)

        messages.append(f"Video frames:")
        for frame_i in frame_list:
            images = Part.from_uri(frame_i, mime_type="image/png")
            messages.append(images)

        messages.append(f"Dialogue:")
        for diag_i in diag_list:
            messages.append(f"{diag_i}.")
        
        messages.append(f"Movie Visual Description:")
        messages.append(f"{Need_AD}.")

        messages.append(query)
        
        # params = {
        #     "model": self.model,
        #     "messages": messages,
        #     "max_tokens": 2048,
        #     "temperature": temperature,
        # }
        
        response = self.client.generate_content(messages)
        return response.text


class MovieSeq_MiniCPMV:
    def __init__(self,
                    model="MiniCPM-V-2_6", image_detail="auto",
                 system_text=None):
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A sequence of photos of characters along with their names.
        2. Keyframes from a video clip.
        3. The corresponding dialogues from the video clip, each associated with a speaker ID.
        4. Manually annotated Movie Visual Description, including the narration for the visual elements in movies.

        Your task is to analyze and associate these inputs, understand the context of the video, and respond to the user's needs accordingly.
        """
            
        # self.headers = {
        # "Content-Type": "application/json",
        # "Authorization": f"Bearer {self.api_key}"
        # }
        # self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

        self.client = AutoModel.from_pretrained('/workspace/wuweijia/Agent/MiniCPM-V/weight/MiniCPM-V-2_6', trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        self.client = self.client.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('/workspace/wuweijia/Agent/MiniCPM-V/weight/MiniCPM-V-2_6', trust_remote_code=True)


    def get_response(self, char_bank, frame_list, diag_list, Need_AD,
                                query,
                                resize=None, temperature=0, detail="auto"):
        content = [self.system_text]
        
            
        for char_name, char_url in char_bank.items():
            char_image = Image.open(char_url).convert('RGB')
            # char_image = encode_image(char_url)
            content.append(f"This is the photo of {char_name}.")
            content.append(char_image)

        # assert len(diag_list) == len(frame_list)
        content.append(f"Video Frame:")
        for frame_i in frame_list:
            frame_image = Image.open(frame_i).convert('RGB')
            content.append(frame_image)
            break

        content.append(f"Dialogue:")
        for diag_i in diag_list:
            content.append(diag_i)
        
        content.append(f"Movie Visual Description:")
        content.append(f"{Need_AD}.")

        content.append(query)

        messages = [{'role': 'user', 'content': content}]
        # msgs = []

        answer = self.client.chat(
            image=None,
            msgs=messages,
            tokenizer=self.tokenizer
        )

        return answer

class MovieSeq_Layout:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail

        # (If there is no character, {} is returned and objects other than the supplied character cannot appear)
        if system_text is None:
            self.system_text = """
        You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting, and the crossponding character name. Your task is to generate the bounding boxes for each character, along with a background prompt describing the scene. The images are of height 1024 and width 2048 and the bounding boxes should not overlap or go beyond the image boundaries. 
        Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object. Make the boxes larger if possible. Do not put objects that are already provided in the bounding boxes into the background prompt. If needed, you can make reasonable guesses. Generate the object descriptions and background prompts in English even if the caption might not be in English. Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format:
        
        Example 1:
        Character: 
            {
                "Barbara Fitts": "Barbara Fitts in red dress",
                "Jim Olmeyer": "Christy Kane wearing a hat",
                "Christy Kane": "Jim Olmeyer in white suit",
            }
        Caption: Barbara Fitts in red dress, Christy Kane wearing a hat, and Jim Olmeyer in white suit are walking near a lake.
        Output: {
            "Characters":
                {
                "Barbara Fitts": "Barbara Fitts in red dress",
                "Jim Olmeyer": "Christy Kane wearing a hat",
                "Christy Kane": "Jim Olmeyer in white suit",
                },
            "plot": "Three people are walking near a lake",
            "box": 
                {
                "Barbara Fitts": [215, 121, 308, 901], 
                "Jim Olmeyer": [600, 20, 420, 1000],
                "Christy Kane": [1119, 88, 387, 900]
                }
            }

        Example 2:
        Character: 
            {
                "Barbara Fitts": "Barbara Fitts in hogwarts school uniform", 
                "Jim Olmeyer": "Jim Olmeyer in hogwarts school uniform"
            }
        Caption:  Barbara Fitts and Jim Olmeyer, both in hogwarts school uniform, holding hands, facing a strong monster, near the castle.
        Output: 
            {
            "Characters":
                {
                "Barbara Fitts": "Barbara Fitts in red dress",
                "Jim Olmeyer": "Christy Kane wearing a hat",
                "Christy Kane": "Jim Olmeyer in white suit",
                },
            "plot": "Two people, both in hogwarts school uniform, holding hands, facing a strong monster, near the castle.",
            "box": 
                {
                "Barbara Fitts": [6, 4, 500, 1020], 
                "Jim Olmeyer": [507, 14, 506, 1010]
                }
            }

        Your task is to analyze and associate the given Character and Caption, and provide the output. 
        The output plot should not include names of people, sequential actions, or detailed descriptions. Keep it as simple as possible.
        The corresponding character description can only include the name and description corresponding to a person, which is a description of a single person without interaction with others.
        Follow the structure below for formatting your output, and can use json.loads to convert it to a dictionary:
        {
            "Characters":
                {
                "Character Name 1": "description for character 1",
                "Character Name 2": "description for character 2",
                },
            "plot": "Description for a simple plot, including the number of people, and do not mention character names.",
            "box": 
            {
            "Character Name 1": [top-left x coordinate, top-left y coordinate, box width, box height], 
            "Character Name 2": [top-left x coordinate, top-left y coordinate, box width, box height]
            }
        }
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
            
        

        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content


class MovieSeq:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A sequence of photos of characters along with their names.
        2. Keyframes from a video clip.
        3. The corresponding dialogues from the video clip, each associated with a speaker ID.
        4. Manually annotated Movie Visual Description, including the narration for the visual elements in movies.

        Your task is to analyze and associate these inputs, understand the context of the video, and respond to the user's needs accordingly.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, char_bank, frame_list, diag_list, Need_AD,
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
            
        for char_name, char_url in char_bank.items():
            char_image = encode_image(char_url)
            messages.append({
                "role": "user",
                "content": [
                    f"This is the photo of {char_name}.",
                    {'image': char_image},
                ],
            })

        # assert len(diag_list) == len(frame_list)
        for frame_i in frame_list:
            frame_image = encode_image(frame_i)
            messages.append({
                "role": "user",
                "content": [
                    {'image': frame_image}
                ],
            })

        for diag_i in diag_list:
            messages.append({
                "role": "user",
                "content": [
                    f"{diag_i}.",
                ],
            })
        
        messages.append({
            "role": "user",
            "content": [
                f"{Need_AD}.",
            ],
        })

        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content



class MovieSeq_Char:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A photos of characters along with the names.

        Your task is to analyze and associate these inputs, and respond to the user's needs accordingly.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, name, char_url, 
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
        
        name = name.split("-")[-1].replace("_"," ")
        char_image = encode_image(char_url)
        messages.append({
            "role": "user",
            "content": [
                f"This is the photo of {name}.",
                {'image': char_image},
            ],
        })

        
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content
    



class GPT4O:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        self.system_text = ""
        # if system_text is None:
        #     self.system_text = """
        # You will be provided with the following inputs:
        # 1. A photos of characters along with the names.

        # Your task is to analyze and associate these inputs, and respond to the user's needs accordingly.
        # """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, video_clip, 
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
        
        # name = name.split("-")[-1].replace("_"," ")
        for image in video_clip:

            char_image = encode_image(image)
            messages.append({
                "role": "user",
                "content": [
                    {'image': char_image},
                ],
            })

        
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content


class MovieSeq_Scene:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with a sequence of keyframes from a video clip. 
        You are required to classify the scene in the video clip and describe it with a short phrase.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, frame_list,frame_list_pre = None, pre_save_path_json=None,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
        
        if frame_list_pre!=None:
            with open(pre_save_path_json, 'r') as file:
                # Read all the lines in the file
                contents = file.read()
            messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": "This is the previous video clip from the same movie and its corresponding scene classification:",},]
            })
            for char_url in frame_list_pre:
                char_image = encode_image(char_url)
                messages.append({
                    "role": "user",
                    "content": [
                        f"scene: {contents}.",
                        {'image': char_image},
                    ],
                })
            messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": "The following input is the current video clip: ",},]
            })
            for frame_i in frame_list:
                frame_image = encode_image(frame_i)
                messages.append({
                    "role": "user",
                    "content": [
                        {'image': frame_image}
                    ],
                })
            messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": "You need to judge the scene of the current video clip based on the scene category of the previous video. If you think they are in the same scene, output the same scene category; otherwise, describe this new scene with a phrase. Note: the scene refers to the location only; please do not include any information about the characters' activities.",},]
            })

        else:
            messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": "The following input is the current video clip: ",},]
            })
            for frame_i in frame_list:
                frame_image = encode_image(frame_i)
                messages.append({
                    "role": "user",
                    "content": [
                        {'image': frame_image}
                    ],
                })
            messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": "Describe this new scene with a phrase. Note: the scene refers to the location only; please do not include any information about the characters' activities.",},]
            })


        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content