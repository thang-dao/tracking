import sys
CENTERNET_PATH = '/home/vietthangtik15/tracking/centernet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
# import _init_paths
from detectors.detector_factory import detector_factory
from opts import opts 

MODEL_PATH = 'centernet/models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector.detector_factory[opt.task](opt)


# video = '/home/viettthangtik15/dataset/input/video_1.mp4'
img = 'images/ori_im.jpg'
ret = detector.run(img)['category_id']

