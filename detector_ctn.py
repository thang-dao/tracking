import sys
import cv2
import glob
CENTERNET_PATH = '/home/vietthangtik15/tracking/centernet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)
from detectors.detector_factory import detector_factory
from opts import opts 

MODEL_PATH = 'centernet/models/ctdet_coco_dla_2x.pth'
TASK = 'ctdet'
opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
detector = detector_factory[opt.task](opt)

# video = '/home/viettthangtik15/dataset/input/video_1.mp4'
dirs = '/home/vietthangtik15/dataset/input/*.jpg'
for img in glob.glob(dirs):
	img = cv2.imread(img, 1)
	ret = detector.run(img)
	count = 0
	# for key, value in ret['results'].items():
	for imcp in ret['results'][1]:
		ids = 1
		x1 = imcp[0]
		y1 = imcp[1] 
		x2 = imcp[2]
		y2 = imcp[3] 
		score = imcp[4]
		if ids == 1:
			count += 1
			imgcrop = img[int(y1):int(y2), int(x1):int(x2)]		
			print(imgcrop.shape)
				# cv2.imwrite('/home/vietthangtik15/dataset/output/' + str(count) +'.jpg', imgcrop)
	print(len(ret['results'][1]), count)
cv2.destroyAllWindows()