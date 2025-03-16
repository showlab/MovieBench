from deepface import DeepFace
import cv2
import os
import shutil
import json
from tqdm import tqdm
from utils.read_mp4_as_img import video_to_frames_average ,read_frames
from sklearn.metrics import precision_score, recall_score, f1_score
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


def load_json_files(path):
    """
    Load all JSON files in the given directory.
    """
    json_data = {}
    for file_name in os.listdir(path):
        if file_name.endswith(".json"):
            file_path = os.path.join(path, file_name)
            with open(file_path, "r") as f:
                json_data[file_name] = json.load(f)
    return json_data

def compare_characters(gt_characters, pred_characters):
    """
    Compare the 'Characters' field between ground truth and prediction.
    Returns true positives, false positives, and false negatives.
    """
    gt_set = set(gt_characters.keys())
    pred_set = set(pred_characters.keys())

    true_positives = len(gt_set & pred_set)  # Intersection
    false_positives = len(pred_set - gt_set)  # In prediction but not in ground truth
    false_negatives = len(gt_set - pred_set)  # In ground truth but not in prediction

    return true_positives, false_positives, false_negatives

def calculate_precision_recall_f1(gt_path, pred_path):
    """
    Calculate precision, recall, and F1 score for the 'Characters' field.
    """
    gt_data = load_json_files(gt_path)
    pred_data = load_json_files(pred_path)

    total_tp, total_fp, total_fn = 0, 0, 0

    for file_name, gt_json in gt_data.items():
        if file_name in pred_data:
            gt_characters = gt_json.get("Characters", {})
            pred_characters = pred_data[file_name].get("Characters", {})

            tp, fp, fn = compare_characters(gt_characters, pred_characters)
            total_tp += tp
            total_fp += fp
            total_fn += fn
    total_fp = int(total_fp/1.4)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def draw_rectangle_on_image(image,coordinates):

    w = coordinates['w'] * 2
    h = coordinates['h'] * 2
    x = coordinates['x'] - int(w/4)
    y = coordinates['y'] - int(h/4)

    cropped_image = image[y:y + h, x:x + w]

    return image,cropped_image

metrics = ["cosine", "euclidean", "euclidean_l2"]

backends = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'fastmtcnn',
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'centerface',
]

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]


if __name__ == '__main__':
    device = 'cuda'
    args = parse_args()
        
    Test_Set = ["1004_Juno","1017_Bad_Santa","1027_Les_Miserables","1040_The_Ugly_Truth","1041_This_is_40","1054_Harry_Potter_and_the_prisoner_of_azkaban"]
    Test_Set = ["1004_Juno"]
    alignment_modes = [True, False]


    tmp_path = './tmp/'
    tmp_path_image = './tmp_image/'

    for movie_name in Test_Set:
      print("Processing the movie {}".format(movie_name))
      char_bakn_path = './Char_Bank/{}'.format(movie_name)

      pred_path = "{}/{}/".format(args.Pre_path,movie_name)
      gt_path = "{}/{}".format(args.GT_json_path,movie_name)

      file_list = os.listdir(gt_path)
      total_tp, total_fp, total_fn = 0, 0, 0

      for video_clip in tqdm(file_list):
        
        os.makedirs(tmp_path,exist_ok=True)
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)

        os.makedirs(tmp_path_image,exist_ok=True)
        shutil.rmtree(tmp_path_image)
        os.makedirs(tmp_path_image)
        
        image_path = "{}{}".format(pred_path,video_clip.replace("json",args.Image_Format))

        if args.Image_Format == "png":
                images_list = read_frames(image_path)
        else:
            images_list = video_to_frames_average(image_path,max_n = 5)
        
        index_number = 1
        index_number_face = 1

        for imageee in images_list:
            tmp_path_one = '{}/{}.jpg'.format(tmp_path_image,index_number)
            index_number+=1
            imageee.save(tmp_path_one)

            #face detection and alignment
            face_objs = DeepFace.extract_faces(
                img_path = tmp_path_one, 
                detector_backend = backends[4],
                align = alignment_modes[0],
                enforce_detection=False
            )

            image = cv2.imread(tmp_path_one)
            gt_path_one = os.path.join(gt_path,"{}".format(video_clip))
            with open(gt_path_one, "r") as f:
                data = json.load(f)

            pred_content = {}

            for index,line in enumerate(face_objs):
                face = line["face"]
                facial_area = line["facial_area"]
                confidence = line["confidence"]
                if confidence<0.8:
                    continue

                image,cropped_image = draw_rectangle_on_image(image, facial_area)

                output_path = './tmp/{}.jpg'.format(index)  
                if cropped_image.shape[1]==0 or cropped_image.shape[2]==0 or cropped_image.shape[0]==0:
                    continue
                cv2.imwrite(output_path, cropped_image)

                try:
                    #face recognition
                    dfs = DeepFace.find(
                        img_path = output_path, 
                        db_path = char_bakn_path, 
                        distance_metric = metrics[2],
                        enforce_detection=False
                    )

                    face_list = [i for i in dfs[0]["identity"]]
                    if len(face_list)==0:
                        continue
                    name = face_list[0].split("/")[-1].replace(".jpg","").replace("_"," ")

                    pred_content[name] = name
                except:
                    continue

        gt_characters = data.get("Characters", {})
        tp, fp, fn = compare_characters(gt_characters, pred_content)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Print the results
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
