
# calculate clip score
# python Metric_1_clip_score.py \
#     --Pre_path /storage/wuweijia/MovieGen/Mix-of-Show/results/inference_StorySalon/ \
#     --GT_json_path /storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test/ \
#     --Format Image\
#     --Resolution 256

# python Metric_1_clip_score.py \
#     --Pre_path /storage/wuweijia/MovieGen/SVD/results \
#     --GT_json_path /storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test/ \
#     --Format Video\
#     --Resolution 256\
#     --Frame_Number 25



# calculate Inception score
# python Metric_2_Inception.py \
#     --Pre_path /storage/wuweijia/MovieGen/Mix-of-Show/results/inference_StorySalon/ \
#     --Format Image\
#     --Resolution 256

# python Metric_2_Inception.py \
#     --Pre_path /storage/wuweijia/MovieGen/SVD/results \
#     --Format Video\
#     --Resolution 256


# calculate aesthetic score
# python Metric_3_aesthetic_score.py \
#     --Pre_path /storage/wuweijia/MovieGen/Mix-of-Show/results/inference_StorySalon/ \
#     --Format Image\
#     --Resolution 100

# python Metric_3_aesthetic_score.py \
#     --Pre_path /storage/wuweijia/MovieGen/SVD/results \
#     --Format Video\
#     --Resolution 100


# calculate FID
# python Metric_4_fid.py \
#     --Pre_path /storage/wuweijia/MovieGen/Mix-of-Show/results/inference_StorySalon/ \
#     --GT_video_path /storage/wuweijia/MovieGen/lsmdc/avi/ \
#     --Format Image\
#     --Resolution 512

# python Metric_4_fid.py \
#     --Pre_path /storage/wuweijia/MovieGen/SVD/results \
#     --GT_video_path /storage/wuweijia/MovieGen/lsmdc/avi/ \
#     --Format Video\
#     --Resolution 512

# calculate FVD
python Metric_6_FVD/run.py \
    --Pre_path /storage/wuweijia/MovieGen/SVD/results \
    --GT_video_path /storage/wuweijia/MovieGen/lsmdc/avi/ \
    --Format Video\
    --Resolution 256\
    --Image_Format mp4\
    --Frame_Number 25

# calculate Character Consistency
# python Metric_5_Character_ID_Consistency.py \
#     --Pre_path /storage/wuweijia/MovieGen/Mix-of-Show/results/multi-concept/ \
#     --GT_json_path /storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test/ \
#     --Format Image\
#     --Resolution 512\
#     --Image_Format png

# python Metric_5_Character_ID_Consistency.py \
#     --Pre_path /storage/wuweijia/MovieGen/SVD/results \
#     --GT_json_path /storage/wuweijia/MovieGen/lsmdc/Annotation_Shot_Desc/Test/ \
#     --Format Video\
#     --Resolution 512\
#     --Image_Format mp4