from pyannote.audio import Pipeline
import torch
import os
from pydub import AudioSegment

speaker_path = "/workspace/wuweijia/MovieGen/lsmdc/GT/Audio"
speaker11_path = "/workspace/wuweijia/MovieGen/lsmdc/GT/Speaker"


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)

pipeline.to(torch.device("cuda"))



movie_list = os.listdir(speaker_path)

for movie_name in movie_list:
    save_movie_p = os.path.join(speaker11_path,movie_name)
    movie_p = os.path.join(speaker_path,movie_name)

    os.makedirs(save_movie_p, exist_ok=True)

    for vide_name in os.listdir(movie_p):
        save_movie_p_one = os.path.join(save_movie_p,vide_name.split(".wa")[0])
        # print(save_movie_p_one)
        # assert False
        os.makedirs(save_movie_p_one, exist_ok=True)

        # apply pretrained pipeline
        diarization = pipeline(os.path.join(movie_p,vide_name))
        # Load the original audio file (assumes your original audio is in WAV format)
        audio = AudioSegment.from_wav(os.path.join(movie_p,vide_name))  # Replace with the path to your WAV file


        # print the result
        idx = 0
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start * 1000  # Convert to milliseconds
            end_time = turn.end * 1000  # Convert to milliseconds

            # Extract the segment of the audio corresponding to the speaker
            speaker_audio = audio[start_time:end_time]

            # Save the segment as a .wav file
            save_movie_p_one_one = os.path.join(save_movie_p_one,vide_name.split(".wa")[0]+"_"+str(idx)+".wav")
            idx+=1
            speaker_audio.export(save_movie_p_one_one, format="wav")

# start=0.2s stop=1.5s speaker_0
# start=1.8s stop=3.9s speaker_1
# start=4.2s stop=5.7s speaker_0
# ...