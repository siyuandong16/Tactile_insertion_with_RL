
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2

class marker_detection:
    def __init__(self):
        self.kernal = self.make_kernal(3, 'circle')

    def make_kernal(self, n, type):
        if type is 'circle':
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernal

    def creat_mask(self, raw_image):
        scale = 2
        m, n = raw_image.shape[1], raw_image.shape[0]
        raw_image = cv2.pyrDown(raw_image).astype(np.float32)
        blur = cv2.GaussianBlur(raw_image, (25, 25), 0)
        blur2 = cv2.GaussianBlur(raw_image, (5, 5), 0)
        diff = blur - blur2
        diff *= 16.0
        diff[diff < 0.] = 0.
        diff[diff > 255.] = 255.
        mask = (diff[:, :, 0] > 25) & (diff[:, :, 2] > 25) & (diff[:, :, 1] >
                                                              120)
        mask = cv2.resize(mask.astype(np.uint8), (m, n))
        # mask = mask * dmask
        mask = cv2.dilate(mask, self.kernal, iterations=1)
        return (1 - mask) * 255

    def find_dots(self, binary_image):
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 1
        params.maxThreshold = 12
        params.minDistBetweenBlobs = 9
        params.filterByArea = True
        params.minArea = 9
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(binary_image.astype(np.uint8))
        return keypoints

    def draw_mask(self, img, keypoints):
        img = np.zeros_like(img[:, :, 0])
        for i in range(len(keypoints)):
            cv2.ellipse(img,(int(keypoints[i].pt[0]), int(keypoints[i].pt[1])),
                        (3, 3), 0, 0, 360, (1), -1)
        return img

    def marker_detection(self, raw_imag):

        img = np.array(raw_imag).astype(np.float32)
        mask1 = self.creat_mask(img)
        keypoints = self.find_dots(mask1)
        mask2 = self.draw_mask(img, keypoints)
        return mask1, mask2
   



if __name__ == "__main__":
    detector = marker_detection()
    im = cv2.imread('/homes/jha/Dropbox/Packing_RL_data/data_newsensor_14/hexagon/166/2.jpg')
    mask1, mask2 = detector.marker_detection(im)

    print("Image size..", mask2.shape)

    cv2.imshow('marker mask1', mask1)
    cv2.imshow('marker mask2', mask2)
    cv2.waitKey(0)

#%%
