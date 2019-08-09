import os
import cv2
import time
import argparse
import numpy as np
import sys
CENTERNET_PATH = '/home/vietthangtik15/tracking/centernet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

from detectors.detector_factory import detector_factory
from opts import opts 

from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes

MODEL_PATH = 'centernet/models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))

class Detector(object):
    def __init__(self, args):
        self.args = args
        args.display = False
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.centernet = detector = detector_factory[opt.task](opt)
        self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint, args.model_name)
        self.class_names = self.yolo3.class_names


    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width,self.im_height))

        assert self.vdo.isOpened()
        return self

    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        

    def detect(self):
        count = 0
        while self.vdo.grab(): 
            start = time.time()
            re, ori_im = self.vdo.retrieve()
            if re == True:
                count += 1
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                im = ori_im 
                # cv2.imwrite("/home/vietthangtik15/dataset/input/" + str(count) + ".jpg", ori_im)
                # bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
                # if bbox_xcycwh is not None:
                #     # select class person 
                #     mask = cls_ids==0
                #     bbox_xcycwh = bbox_xcycwh[mask]
                #     bbox_xcycwh[:,3:] *= 1.2
                #     cls_conf = cls_conf[mask]
                #     outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                #     if len(outputs) > 0:
                #         bbox_xyxy = outputs[:,:4]
                #         identities = outputs[:,-1]
                #         ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)


                ret = self.centernet.run(ori_im)
                confidences = []
                if ret['results'] is not None:
                    for confi in ret['results'][1]:
                        confidences.append(confi[4])
                    
                    for imcp in ret['results'][1]:
                        ids = 1
                        x1 = imcp[0]
                        y1 = imcp[1] 
                        x2 = imcp[2]
                        y2 = imcp[3] 
                        score = imcp[4]
                        print('correct', x1, y1, x2, y2)
                        if ids == 1:
                            count += 1
                            imgcrop = ori_im[int(y1):int(y2), int(x1):int(x2)]		
                            print(imgcrop.shape)    

                    outputs = self.deepsort.update(ret['results'][1], confidences, im)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:,:4]
                        identities = outputs[:,-1]
                        ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)
                end = time.time()
                print("time: {}s, fps: {}".format(end-start, 1/(end-start)))

                if self.args.save_path:
                    self.output.write(ori_im)
            else:
                break

            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="/home/vietthangtik15/dataset/input/video_1.mp4")
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/patchnet.pth")

    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--model_name", type=str, default='patchnet')
    parser.add_argument("--save_path", type=str, default="/home/vietthangtik15/dataset/output/demo.avi")
    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
