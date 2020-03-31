import numpy as np
import time
import matplotlib.pyplot as plt
import math, cv2, os
import copy
from scipy.interpolate import griddata


class marker_flow:
    def __init__(self):
        self.kernel1 = self.make_kernel(3, 'circle')
        self.kernel2 = self.make_kernel(45, 'circle')
        self.scale = 1
        self.refresh1 = False
        self.refresh2 = False
        self.marker_thre = 100

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.33, 0.33, 0.34])

    def make_kernel(self, n, type):
        if type is 'circle':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (n, n))
        return kernel

    def defect_mask(self, im_cal):
        pad = 60
        var0 = 60  #left up
        var1 = 60  # right up
        var2 = 65  # right down
        var3 = 60  # left down
        im_mask = np.ones((im_cal.shape))
        triangle0 = np.array([[0, 0], [var0, 0], [0, var0]])
        triangle1 = np.array([[im_mask.shape[1] - var1, 0],
                              [im_mask.shape[1], 0], [im_mask.shape[1], var1]])
        triangle2 = np.array([[im_mask.shape[1] - var2, im_mask.shape[0]], [im_mask.shape[1], im_mask.shape[0]], \
            [im_mask.shape[1], im_mask.shape[0]-var2]])
        triangle3 = np.array([[0, im_mask.shape[0]],
                              [0, im_mask.shape[0] - var3],
                              [var3, im_mask.shape[0]]])
        color = [0]  #im_mask
        cv2.fillConvexPoly(im_mask, triangle0, color)
        cv2.fillConvexPoly(im_mask, triangle1, color)
        cv2.fillConvexPoly(im_mask, triangle2, color)
        cv2.fillConvexPoly(im_mask, triangle3, color)
        im_mask[:pad, :] = 0
        im_mask[-pad:, :] = 0
        im_mask[:, :pad] = 0
        im_mask[:, -pad:] = 0
        return im_mask

    def make_thre_mask(self, im_cal):
        thre_image = np.zeros(im_cal.shape, dtype=np.uint8)
        previous_mask = np.zeros(im_cal.shape, dtype=np.uint8)
        for i in range(10, 80, 30):
            _, mask = cv2.threshold(im_cal.astype(np.uint8), i, 255,
                                    cv2.THRESH_BINARY_INV)
            mask_expand = cv2.dilate(mask, self.kernel2, iterations=1)
            mask_erode = cv2.erode(mask_expand, self.kernel2, iterations=1)
            thre_image += (mask_erode - previous_mask) / 255 * i
            previous_mask = mask_erode
            # cv2.imshow('threshold', thre_image)
            # cv2.waitKey(0)
        thre_image += (np.ones(im_cal.shape, dtype=np.uint8) -
                       previous_mask / 255) * 80 + 10

        return thre_image

    def creat_mask_2(self, raw_image, dmask):
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
        mask = mask * dmask
        mask = cv2.dilate(mask, self.kernel1, iterations=1)
        return (1 - mask) * 255

    def find_dots(self, binary_image):
        # down_image = cv2.resize(binary_image, None, fx=2, fy=2)
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

    def flow_calculate_in_contact(self, keypoints2, x_initial, y_initial,
                                  u_ref, v_ref):
        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired = [], [], [], [], [], [], [], []

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i].pt[0] / self.scale)
            y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)
        index_list = []

        for i in range(x2.shape[0]):

            distance = list(((np.array(x_initial) - x2[i])**2 +
                             (np.array(y_initial) - y2[i])**2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - x_initial[min_index]
            v_temp = y2[i] - y_initial[min_index]
            shift_length = np.sqrt(u_temp**2 + v_temp**2)
            # print 'length',shift_length

            if shift_length < 12:
                # print xy2.shape,min_index,len(distance)
                x1_paired.append(x_initial[min_index] - u_ref[min_index])
                y1_paired.append(y_initial[min_index] - v_ref[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + u_ref[min_index])
                v.append(v_temp + v_ref[min_index])
                index_list.append(self.index_list[min_index])

                if shift_length > 7:
                    refresh = True
                else:
                    refresh = False

        return x1_paired, y1_paired, x2_paired, y2_paired, u, v, refresh, index_list

    def flow_calculate_global(self, keypoints2, x_initial, y_initial, u_ref,
                              v_ref):
        x2, y2, u, v, x1_paired, y1_paired, x2_paired, y2_paired  = [], [], [], [], [], [], [], []
        x1_return, y1_return, x2_return, y2_return, u_return, v_return = [],[],[],[],[],[]

        for i in range(len(keypoints2)):
            x2.append(keypoints2[i].pt[0] / self.scale)
            y2.append(keypoints2[i].pt[1] / self.scale)

        x2 = np.array(x2)
        y2 = np.array(y2)
        index_list = []

        for i in range(x2.shape[0]):
            distance = list(((np.array(x_initial) - x2[i])**2 +
                             (np.array(y_initial) - y2[i])**2))
            if len(distance) == 0:
                break
            min_index = distance.index(min(distance))
            u_temp = x2[i] - x_initial[min_index]
            v_temp = y2[i] - y_initial[min_index]
            shift_length = np.sqrt(u_temp**2 + v_temp**2)
            # print 'length',shift_length
            if shift_length < 12:
                x1_paired.append(x_initial[min_index] - u_ref[min_index])
                y1_paired.append(y_initial[min_index] - v_ref[min_index])
                x2_paired.append(x2[i])
                y2_paired.append(y2[i])
                u.append(u_temp + u_ref[min_index])
                v.append(v_temp + v_ref[min_index])
                index_list.append(self.index_list[min_index])

                # del x_initial[min_index], y_initial[min_index], u_ref[
                #     min_index], v_ref[min_index]

        x1_return = np.array(x1_paired)
        y1_return = np.array(y1_paired)
        x2_return = np.array(x2_paired)
        y2_return = np.array(y2_paired)
        u_return = np.array(u)
        v_return = np.array(v)

        return x1_return, y1_return, x2_return, y2_return, u_return, v_return, \
            list(x2_paired), list(y2_paired), np.array(x2_paired), np.array(y2_paired), index_list
        # return x1_paired,y1_paired,x2_paired,y2_paired,u,v

    def dispOpticalFlow(self, im_cal, x, y, u, v, name):
        # mask = np.zeros_like(im_cal)
        mask2 = np.zeros_like(im_cal)
        amf = 3
        x = np.array(x).astype(np.int16)
        y = np.array(y).astype(np.int16)
        for i in range(u.shape[0]):  #self.u_sum

            mask2 = cv2.line(mask2,
                             (int(x[i] + u[i] * amf), int(y[i] + v[i] * amf)),
                             (x[i], y[i]), [0, 120, 120], 2)

        img = cv2.add(im_cal, mask2)

        cv2.imshow(name, img.astype(np.uint8))
        cv2.waitKey(0)

    def main(self, raw_imag, first_image):

        if first_image:

            imgwc = np.array(raw_imag).astype(np.float32)
            self.im_slipsign = np.zeros(imgwc.shape)
            cv2.putText(self.im_slipsign, 'Slip', (210, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            im_gray = self.rgb2gray(imgwc)  #.astype(np.uint8)
            self.dmask1 = self.defect_mask(im_gray)
            final_image = self.creat_mask_2(imgwc, self.dmask1)
            keypoints = self.find_dots(final_image)
            self.u_sum1 = np.zeros(len(keypoints))
            self.v_sum1 = np.zeros(len(keypoints))
            self.u_addon1 = list(self.u_sum1)
            self.v_addon1 = list(self.v_sum1)
            self.x1_last1 = []
            self.y1_last1 = []
            for i in range(len(keypoints)):
                self.x1_last1.append(keypoints[i].pt[0] / self.scale)
                self.y1_last1.append(keypoints[i].pt[1] / self.scale)
            self.x_iniref1 = list(self.x1_last1)
            self.y_iniref1 = list(self.y1_last1)
            self.absmotion1 = 0
            self.index_list = range(len(self.x1_last1))
            return np.array(self.x1_last1), np.array(
                self.y1_last1), np.zeros_like(self.x1_last1), np.zeros_like(
                    self.x1_last1), np.array(self.index_list)

        else:
            imgwc = np.array(raw_imag).astype(np.float32)
            final_image = self.creat_mask_2(imgwc, self.dmask1)

            if self.refresh1:
                keypoints = self.find_dots(final_image)
                x1, y1, x2, y2, u, v, self.x_iniref1, self.y_iniref1, self.u_addon1, self.v_addon1, index_list\
                    = self.flow_calculate_global(keypoints, list(self.x_iniref1), list(self.y_iniref1), \
                        list(self.u_addon1), list(self.v_addon1))
                self.refresh1 = False
            else:
                keypoints = self.find_dots(final_image)
                x1, y1, x2, y2, u, v, self.refresh1, index_list = self.flow_calculate_in_contact(
                    keypoints, list(self.x_iniref1), list(self.y_iniref1),
                    list(self.u_addon1), list(self.v_addon1))

            return np.array(x2), np.array(y2), np.array(u), np.array(
                v), np.array(index_list)


if __name__ == "__main__":

    mf = marker_flow()
    path = '/media/mcube/SERVER_HD/siyuan/policy_finetune/'
    ob = 'circle'
    num = 10
    x_min, x_max, y_min, y_max = 62, 366, 63, 256  #marker tracking region
    gap = 10  #pixel gap between each measurement

    for i in range(12):  #read the 12 images during contact period
        img = cv2.imread(path + ob + '/' + str(num) + '/' + str(i) + '.jpg')
        x, y, u, v, index_list = mf.main(img, i == 0)

        if i == 0:
            x_ref, y_ref = copy.deepcopy(x), copy.deepcopy(y)
            x_paired, y_paired = copy.deepcopy(x), copy.deepcopy(y)
            # x_min, x_max = int(np.min(x_ref)), int(np.max(x_ref)) + 1
            # y_min, y_max = int(np.min(y_ref)), int(np.max(y_ref)) + 1
            x_grid, y_grid = np.meshgrid(range(x_min, x_max, 10),
                                         range(y_min, y_max, 10))
            u_grid = np.zeros_like(x_grid)
            v_grid = np.zeros_like(x_grid)
        else:
            x_paired, y_paired = x_ref[index_list], y_ref[index_list]

            points = np.squeeze(np.dstack((x_paired.T, y_paired.T)))
            values = np.squeeze(np.dstack((u.T, v.T)))

            uv = griddata(points,
                          values, (x_grid.flatten(), y_grid.flatten()),
                          method='linear')
            uv[np.isnan(uv)] = 0.
            u_grid = uv[:, 0]
            v_grid = uv[:, 1]

        u_grid_image = np.reshape(u_grid, x_grid.shape)
        v_grid_image = np.reshape(v_grid, x_grid.shape)

        cv2.imshow('u', ((u_grid_image + 10) * 12).astype(np.uint8))
        cv2.imshow('v', ((v_grid_image + 10) * 12).astype(np.uint8))
        mf.dispOpticalFlow(img, x_grid.flatten(), y_grid.flatten(),
                           u_grid.flatten(), v_grid.flatten(), 'flow')
