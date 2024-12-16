from moviepy.editor import VideoFileClip
import os

video_path = "/workspace/wuweijia/MovieGen/lsmdc/avi/"
audio_path = "/workspace/wuweijia/MovieGen/lsmdc/GT/Audio"

movie_list = os.listdir(video_path)

for movie_name in movie_list:
    save_movie_p = os.path.join(audio_path,movie_name)
    movie_p = os.path.join(video_path,movie_name)

    os.makedirs(save_movie_p, exist_ok=True)

    for vide_name in os.listdir(movie_p):
        # Load the video file
        video = VideoFileClip(os.path.join(movie_p,vide_name))

        # Extract the audio
        audio = video.audio

        # Write the audio to a WAV file
        audio.write_audiofile(os.path.join(save_movie_p,vide_name.replace("avi","wav")))

        # Close the video file
        video.close()
