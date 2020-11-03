"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""
import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import numpy as np

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO

#kalman filter
from utils import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import nn_matching
from deep_sort.sort import *

WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=1,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str,default='yolov3-416',
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    # parser.add_argument(
    #     '-vr', '--video_record', type=bool, default=False,
    #     help='record video as *.avi')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    
    tic = time.time()
    
    """
    Video save
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('result_3.avi', fourcc, 50.0, (1920, 1080), True)
    
    
    """
    Initialize tracker
    --max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    --nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    """
    # max_cosine_distance=0.2
    # nn_budget=100
    # metric = nn_matching.NearestNeighborDistanceMetric(
    #     "cosine", max_cosine_distance, nn_budget)
    # print("Metric is :",metric)
    # tracker = Tracker(metric)
    # results = []
    """
    max_age: Maximum number of frames to keep alive a track without associated detections
    min_hits: Minimum number of associated detections before track is initialised.
    iou_threshold: Minimum IOU for match.
    """
    mot_tracker=Sort(max_age=50, 
                       min_hits=1,
                       iou_threshold=0.3)
    
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        img = cam.read()
        if img is None:
            break

        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        
        dets=[]
        for i in range(len(boxes)):
            dets.append(np.append(np.asarray(boxes[i],dtype=np.float),confs[i]))
        if(len(boxes)>0):
            dets_array=np.array(dets)
        else:
            dets_array=np.empty((0,5))

        track_bbs_ids = mot_tracker.update(dets_array)
        print("After pridiction:",track_bbs_ids)
        track_bbs_ids=[track_bb_id[:4] for track_bb_id in track_bbs_ids]
        boxes=np.array(track_bbs_ids,dtype=int)
        print(np.array(track_bbs_ids,dtype=int))

        
        img = vis.draw_bboxes(img, boxes, confs, clss)
        img = show_fps(img, fps)
        cv2.imshow(WINDOW_NAME, img)
        out.write(img)
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
    cam.release()
    out.release()
    cv2.destroyAllWindows()


"""
     Create detections for given frame index from the raw detection matrix.
                                Parameters
"""
def create_detections(boxes, confs):
    """
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.
    """
    detection_list=[]
    for index in range(len(boxes)):
        # print("bbox:",boxes[index])
        # print("confidence:",confs[index])
        detection_list.append(Detection(boxes[index], confs[index],None))
    return detection_list



def main():
    args = parse_args()
    """
    python3 trt_yolo.py --image /home/zq/zq/git-space/tensorrt_about/tensorrt_demos/doc/BLUR20200422143616.jpg -m yolov3-416
    python3 trt_yolo_with_screen.py --video /home/zq/Videos/20201022.flv -m yolov3-416
    """
    """
    check the .trt and number of category num
    """
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/darknet/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)
    """
    check the camera state
    """
    print(args)
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
    yolo_dim = args.model.split('-')[-1]

    if 'x' in yolo_dim:
        dim_split = yolo_dim.split('x')
        if len(dim_split) != 2:
            raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
        w, h = int(dim_split[0]), int(dim_split[1])
    else:
        h = w = int(yolo_dim)
    if h % 32 != 0 or w % 32 != 0:
        raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
    # print("h,w",(h,w))
    trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)

    open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

