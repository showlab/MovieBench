## Metric

### Install the environment.
```bash
cd Metric
conda create -n MovieBench python=3.8
conda activate MovieBench
pip install -r requirements.txt
```

Download clip-vit-large-patch14 for Aesthetic Score Evaluation
```bash
mkdir weights
cd weights
git lfs install
git clone https://huggingface.co/openai/clip-vit-large-patch14
```

### Data Format
The format of the organizational data is as follows:

```
├── MovieBench/ 
|   ├── Annotation_Shot_Desc_11.15_V2_160movies
|   |   ├—— 0001_American_Beauty
|   |   ├—— 0002_As_Good_As_It_Gets
|   |   |   ├—— 0002_As_Good_As_It_Gets_00.00.43.459-00.00.43.636.json
|   |   |   ├—— ...
|   ├── ├—— Ground Truth for video (Avi)
|   |   ├—— 0001_American_Beauty
|   |   ├—— 0002_As_Good_As_It_Gets
|   |   |   ├—— 0002_As_Good_As_It_Gets_00.00.43.459-00.00.43.636.avi
|   |   |   ├—— ...
|   ├── Pre_path(image)
|   |   ├—— 1004_Juno
|   |   ├—— 1017_Bad_Santa
|   |   |   ├—— 1017_Bad_Santa_00.00.05.159-00.00.09.689.jpg
|   |   |   ├—— 1017_Bad_Santa_00.00.47.519-00.00.49.479.jpg
|   |   |   ├—— ...
|   ├── Pre_path(video)
|   |   ├—— 1004_Juno
|   |   ├—— 1017_Bad_Santa
|   |   |   ├—— 1017_Bad_Santa_00.00.05.159-00.00.09.689.mp4
|   |   |   ├—— 1017_Bad_Santa_00.00.47.519-00.00.49.479.mp4
|   |   |   ├—— ...
```


### Metric: Clip Score

#### Calculates the clip score between key frames


```bash
python Metric_1_clip_score.py \
     --Pre_path "path for the predicted result" \
     --GT_json_path "path for the groud truth ( Annotation_Shot_Desc_11.15_V2_160movies)" \
     --Format Image\
     --Resolution 256
```

#### Calculates the clip score between videos


```bash
python Metric_1_clip_score.py \
    --Pre_path "path for the predicted result" \
    --GT_json_path "path for the groud truth ( Annotation_Shot_Desc_11.15_V2_160movies)" \
    --Format Video\
    --Resolution 256\
    --Frame_Number 25
```


### Metric: Inception Score

#### Calculates the Inception Score between key frames


```bash
python Metric_2_Inception.py \
    --Pre_path "path for the predicted result" \
    --Format Image\
    --Resolution 256
```

#### Calculates the Inception Score between videos


```bash
python Metric_2_Inception.py \
    --Pre_path "path for the predicted result" \
    --Format Video\
    --Resolution 256
```


### Metric: Aesthetic Score

#### Calculates the Aesthetic Score between key frames

```bash
python Metric_3_aesthetic_score.py \
    --Pre_path "path for the predicted result" \
    --Format Image\
    --Resolution 100
```

#### Calculates the Inception Score between videos

```bash
python Metric_3_aesthetic_score.py \
    --Pre_path "path for the predicted result" \
    --Format Video\
    --Resolution 100
```

### Metric: calculate FID

#### Calculates the FID between key frames

```bash
python Metric_4_fid.py \
    --Pre_path /"path for the predicted result" \
    --GT_video_path "Ground Truth for video" \
    --Format Image\
    --Resolution 512
```

#### Calculates the Inception Score between videos

```bash
python Metric_4_fid.py \
    --Pre_path "path for the predicted result" \
    --GT_video_path "Ground Truth for video" \
    --Format Video\
    --Resolution 512
```

### Metric: FVD

```bash
python Metric_6_FVD/run.py \
    --Pre_path "path for the predicted result" \
    --GT_video_path "Ground Truth for video"  \
    --Format Video\
    --Resolution 512\
    --Image_Format mp4\
    --Frame_Number 25
```


### Metric: calculate Character Consistency

#### Calculates the Character Consistency between key frames

```bash
python Metric_5_Character_ID_Consistency.py \
    --Pre_path "path for the predicted result" \
    --GT_json_path "path for the groud truth ( Annotation_Shot_Desc_11.15_V2_160movies)"  \
    --Format Image\
    --Resolution 512\
    --Image_Format png
```

#### Calculates the Inception Score between videos

```bash
python Metric_5_Character_ID_Consistency.py \
    --Pre_path "path for the predicted result" \
    --GT_json_path "path for the groud truth ( Annotation_Shot_Desc_11.15_V2_160movies)"  \
    --Format Video\
    --Resolution 512\
    --Image_Format mp4
```