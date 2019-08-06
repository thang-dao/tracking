import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image

from .model import Net
from .patchnet import patchnet
from torchtools import load_pretrained_weights

class Extractor(object):
    def __init__(self, model_path, model_name, use_cuda=True):
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        # state_dict = torch.load(model_path)['net_dict']
        # state_dict = torch.load(model_path)
        self.size = (64, 128)
        print("Loading weights from {}... Done!".format(model_path))
        if model_name == "darknet":
            self.net = Net(reid=True)
            self.norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif model_name == "patchnet":
            self.net = patchnet()
            self.transform = transforms.Compose([
                transforms.Resize((384, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.net = load_pretrained_weights(self.net, model_path)
        # self.net.load_state_dict(torch.load(model_path)['state_dict'])
        # self.net.eval()
        self.net.to(self.device)


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            imgCrop = image[int(y1):int(y2), int(x1):int(x2)]
            imgCrop = transform(Image.fromarray(imgCrop).convert("RGB"))
            imgCrop = imgCrop.unsqueeze(0)
            imgCrops.append(imgCrop)

        imgCrops = torch.cat(imgCrops, 0)
        imgCrops
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def extract_reid_features(self, tlbrs, image):
        imgCrops = []
        for box in tlbrs:
            x1, x2, y1, y2 = box
            print(x1, y1, x2, y2)
            imgCrop = image[int(y1):int(y2), int(x1):int(x2)]
            print(imgCrop.shape)
            imgCrop = self.transform(Image.fromarray(imgCrop).convert("RGB"))
            imgCrop = imgCrop.unsqueeze(0)
            imgCrops.append(imgCrop)

        imgCrops = torch.cat(imgCrops, 0)
        imgCrops = imgCrops.cuda()
        return imgCrops

    # def __call__(self, im_crops):
    #     im_batch = extract_reid_features(self.norm, image, tlbrs)
    #     # im_batch = self._preprocess(im_crops)
    #     with torch.no_grad():
    #         im_batch = im_batch.to(self.device)
    #         # print('im batch', im_batch.shape)
    #         features = self.net(im_batch) 
    #     return features.cpu().numpy()

    def __call__(self, bbox_xywh, image):
        im_batch = self.extract_reid_features(bbox_xywh, image)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            # print('im batch', im_batch.shape)
            features = self.net(im_batch)
        return features.cpu().numpy()



if __name__ == '__main__':
    img = cv2.imread("/home/vietthangtik15/tracking/images/1.jpg")[:,:,(2,1,0)]
    extr = Extractor("/home/vietthangtik15/tracking/deep_sort/deep/checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

