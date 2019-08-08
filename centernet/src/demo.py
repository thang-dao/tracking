from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import time 

from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  out_video = cv2.VideoWriter('/home/vietthangtik15/dataset/output/out_video_gtopia.avi',fourcc, 20.0, (640,480),True)
  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    start = time.time()
    num_frame = 0
    while True:
        re, img = cam.read()
        if re == True:
          # img = cv2.flip(img, 0)
          num_frame += 1
          # out_video.write(img)
          ret = detector.run(img, out_video)
          
          time_str = ''
          for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
          print(time_str)
          # if cv2.waitKey(1) == 27:
              # return  # esc to quit
        else:
            break
    out_video.release()       
    end = time.time()
    seconds = end  - start
    print('FPS', num_frame/seconds)
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
