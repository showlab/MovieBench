import time
import os
import json
import imageio

import requests
import argparse
from tqdm import tqdm

from moviepy.editor import VideoFileClip
from utils import encode_image
import numpy as np
from movieseq import MovieSeq

import shutil
import openai
from openai import OpenAI
import cv2
import whisperx
import pandas as pd
import json

def prepare_dialogue(video_path,model,diarize_model,model_a,metadata):
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=32)
    result_a = whisperx.align(result["segments"], model_a, metadata, audio,
                              return_char_alignments=False, device='cuda')
    
    diarize_segments = diarize_model(audio)
    result_id = whisperx.assign_word_speakers(diarize_segments, result_a)
    return result_id

def prepare_context(vid_url, cut_clip,model,diarize_model,model_a,metadata, n=6):
    # Context -- Dialogue / Subtitles
    vid_asr_id = prepare_dialogue(vid_url,model,diarize_model,model_a,metadata)

    # Video sampling and prepare keyframes & dialogues
    video = VideoFileClip(vid_url)
    duration = video.duration

    prev_speaker = None
    current_segment = []
    all_segments = []
    start_timestamps = []

    for x in vid_asr_id['segments']:
        if 'speaker' in x:
            if prev_speaker is None:
                prev_speaker = x['speaker']
                start_timestamps.append(x['start'])
            
            if x['speaker'] == prev_speaker:
                current_segment.append(f"{x['text']}")
            else:
                all_segments.append(f"{prev_speaker}: {' '.join(current_segment)}")
                current_segment = [f"{x['text']}"]
                prev_speaker = x['speaker']
                start_timestamps.append(x['start'])

    if current_segment:
        all_segments.append(f"{prev_speaker}: {' '.join(current_segment)}")

    asr_list = [all_segments[i:i + cut_clip] for i in range(0, len(all_segments), cut_clip)]
    timestamps = [start_timestamps[i] for i in range(0, len(start_timestamps), cut_clip)]

    time_list = []
    diag_list = []
    for i, clips in enumerate(asr_list):
        time_list.append(timestamps[i])
        diag_list.append(f" ".join(clips))

    output_dir = 'frames'
    os.makedirs(output_dir, exist_ok=True)

    # duration = video.duration
    # interval = (duration-1) / (n - 1)
    # frame_list_11 = [video.get_frame(i * int(interval)) for i in range(n)]

    total_frames = video.reader.nframes
    frame_indices = np.linspace(0, total_frames - 1, n, dtype=int)
    frame_list_11 = []
    for i in frame_indices:
        frame = video.get_frame(i / video.fps) 
        frame_list_11.append(frame) 

    frame_list = []
    for timestamp,frame in enumerate(frame_list_11):
        frame_path = os.path.join(output_dir, f"{timestamp:.1f}.jpg")
        imageio.imwrite(frame_path, frame)
        frame_list.append(frame_path)

    return diag_list, frame_list


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-a', '--openaikey',
        type=str,
        default=None,
        help='OPENAI API KEY'
    )
    parser.add_argument(
        '-hf', '--hftoken',
        type=str,
        default=None,
        help='HF_TOKEN'
    )
    parser.add_argument(
        '-v', '--vid_url',
        type=str,
        default=None,
        help='Path for video'
    )
    parser.add_argument(
        '-c', '--char_bank_path',
        type=str,
        default=None,
        help='Path for character bank'
    )

    parser.add_argument(
        '-u', '--csv_audio_description_path',
        type=str,
        default=None,
        help='Path for video'
    )
    args = parser.parse_args()

    # Set up OPENAI KEY and Models
    os.environ['OPENAI_API_KEY'] = args.openaikey
    
    openai.api_key = os.getenv("OPENAI_API_KEY")
    char_size = 128


    # Please provide Huggingface tokens to access speaker-identify model
    HF_TOKEN = args.hftoken


    # load whisperx model
    model = whisperx.load_model('large-v3', device='cuda')
    model_a, metadata = whisperx.load_align_model(language_code='en', device='cuda')
    diarize_model = whisperx.DiarizationPipeline(model_name='pyannote/speaker-diarization-3.1', use_auth_token=HF_TOKEN, device='cuda')

    # vid_url
    vid_url = args.vid_url
    char_bank_path = args.char_bank_path
    csv_path = args.csv_audio_description_path 

    # proceed the video into # clips
    cut_clip = 4

    movieseq = MovieSeq()

    diag_list, frame_list = prepare_context(vid_url, cut_clip, model, diarize_model, model_a, metadata)

    output_dir = './frames/Char_Bank'
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)


    # Audio Description
    AD_desc = {}
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    for index, row in df.iterrows():
        vid_name, AD = row[0].split("\t")
        AD_desc[vid_name] = AD

    
    try:
        name = vid_url.split("/")[-1].replace(".avi","")
        Need_AD = AD_desc[name]
    except:
        Need_AD = ""

    # Context -- Character / Images
    char_bank = {}
    for character_name in os.listdir(char_bank_path):
        if character_name == ".DS_Store":
            continue

        name = character_name.split("-")[1].replace("_"," ")
        photo_path = os.path.join(char_bank_path,character_name,"best.jpg")

        image = cv2.imread(photo_path)

        height, width = image.shape[:2]

        # Determine the scale factor based on the shorter side
        if height < width:
            new_height = char_size
            new_width = int(width * (char_size / height))
        else:
            new_width = char_size
            new_height = int(height * (char_size / width))

        # Resize the image with the calculated dimensions
        resized_image = cv2.resize(image, (new_width, new_height))
        cv2.imwrite(os.path.join(output_dir,name+".jpg"), resized_image)

        # shutil.copy(photo_path, os.path.join(output_dir,name+".jpg"))
        char_bank[name] = os.path.join(output_dir,name+".jpg")


    print("--------Char Bank-------")
    print(char_bank)

    print("--------Diag Bank-------")
    print(diag_list)

    print("--------Need AD-------")
    print(Need_AD)

    query = """
            Based on the provided video frames, character names and stills, as well as the corresponding narration descriptions, the following content is summarized:
            1. Which characters appear in the video clip, and provide a concise description of each. Note: Describe the character appearance and behavior in the video clip, not the still image.  Only need include the characters from the provided character bank. If none are present, output an empty dictionary {} for character.
            2. Summarize up to three style elements for the video clip, which may include aspects such as setting, background, style, mood, etc.
            3. Provide a detailed Visual Descriptions/Plot for the video clip, including which characters are involved and what activities they are doing. Please be objective and avoid making guesses. Note: Focus primarily on the visual content in the video clip
            4. Provide a detailed background description for the video clip, including objects present in the background, and other relevant details.
            5. Provide a detailed description for the Camera Motion of the video clip.
            Follow the structure below for formatting your output, and can use json.loads to convert it to a dictionary:
            {
                "Characters":
                {
                    "Character Name 1": "Description for appearance and behavior of Character 1, within 30 words",
                    "Character Name 2": "Description for appearance and behavior of Character 2, within 30 words", 
                },
                "Style Elements":
                [
                    "Element 1", "Element 2", "Element 3"
                ],
                "Plot":"A concise summary focusing on the main event or emotion, within 80 words",
                "Background Description":"A concise summary focusing on the main event or emotion, within 40 words",
                "Camera Motion":"A concise summary focusing camera motion, within 30 words."
            }
            """

    text = movieseq.get_response(char_bank, frame_list, diag_list, Need_AD, query)
    text = text.replace("```json","").replace("```","")
    print(text)
    text_json = json.loads(text)
    with open("./test.json", 'w') as file:
        json.dump(text_json, file, indent=4)  


if __name__ == '__main__':
    main()
