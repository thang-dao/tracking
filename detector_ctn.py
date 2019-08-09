import sys
import cv2
CENTERNET_PATH = '/home/vietthangtik15/tracking/centernet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
# import _init_paths
from detectors.detector_factory import detector_factory
from opts import opts 

MODEL_PATH = 'centernet/models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)


# video = '/home/viettthangtik15/dataset/input/video_1.mp4'
img = 'centernet/images/17790319373_bd19b24cfc_k.jpg'
ret = detector.run(img)
for key, value in ret['results'].items():
	id = key
	x1 = value[0]
	# y1 = value[1] 
	# x2 = value[2]
	# y2 = value[3] 
	# score = value[4]
	print(id, x1)
	# imgcrop = img[int(y1):int(y2), int(x1):int(x2)]
	# cv2.imwrite('/home/dataset/output/' + str(x1) +'jpg', imgcrop)
# cv2.destroyAllWindows()