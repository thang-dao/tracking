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
image = 'centernet/images/17790319373_bd19b24cfc_k.jpg'
img = cv2.imread(image, 1)
ret = detector.run(img)
for key, value in ret['results'].items():
	for imcp in value:
		ids = key
		x1 = imcp[0]
		y1 = imcp[1] 
		x2 = imcp[2]
		y2 = imcp[3] 
		score = imcp[4]
		print(ids)
		if score > 0.3 and id == 0:
			imgcrop = img[int(y1):int(y2), int(x1):int(x2)]		
			cv2.imwrite('/home/vietthangtik15/dataset/output/' + str(x1) +'.jpg', imgcrop)
cv2.destroyAllWindows()