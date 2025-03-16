import cv2
from PIL import Image

def read_frames(video_path,size = False):
    imgs = []

    pil = Image.open(video_path)
    if size:
        pil = pil.resize((size[0], size[1]))
    imgs.append(pil)

    

    return imgs

def video_to_frames_average(video_path, size=None, max_n=None):
    imgs = []
    video = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sample interval based on max_n
    if max_n and max_n < total_frames:
        interval = total_frames // max_n
    else:
        interval = 1  # If max_n is not provided or more than total frames, capture every frame
    
    count = 0
    frame_count = 0
    success, image = video.read()

    while success:
        # Only process frames at specified intervals
        if count % interval == 0:
            pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
            if size:
                pil = pil.resize(size)
            imgs.append(pil)
            frame_count += 1
            
            # Stop if we've reached max_n frames
            if max_n and frame_count >= max_n:
                break

        success, image = video.read()
        count += 1

    video.release()
    return imgs

def video_to_frames(video_path, size = None, max_n=None):
    imgs = []
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    sample_rate = int(fps / 3)  # 每秒采样3帧

    success, image = video.read()
    count = 0
    
    pil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), 'RGB')
    if size:
        pil = pil.resize(size)
    imgs.append(pil)

    while success:
        success, image = video.read()
        
        
        if success:
            if count % sample_rate == 0:
                pil = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), 'RGB')
                if size:
                    pil = pil.resize(size)

            if max_n:
                if count>max_n:
                    break
            imgs.append(pil)

        count += 1
    video.release()

    return imgs


if __name__ == '__main__':
    x = video_to_frames("/mnt/nas/share/home/canyu/Diffusion/generative-models/diffusers_svd/78_cutoff10_noinject2.mp4")
    print(x)