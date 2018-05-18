# -*- coding: utf-8 -*-
# Author = ZRHonor

# import os
import cv2
import data
import seg_pneum
import sys

root_path = '../data/5/'
cv2.waitKey()
pneum_area = []
lung_area = []
# for i in range(1, 70):
#     # print(i)
#     path = root_path + '{0}.dcm'.format(i)
#     img = data.load_dcm(path)
#     # flag, pneum, pneum_contours, ratio = seg_pneum.seg_pneum(img)
#     flag, pneum, pneum_contours, pneum_s, lung_s = seg_pneum.seg_pneum(img)
#     pneum_area.append(pneum_s)
#     lung_area.append(lung_s)
#     # print(flag, ratio)
#     cv2.waitKey(100)
# # print('pneum: ', pneum_area)
# # print('lung: ', lung_area)
# total_pneum_area = sum(filter(None, pneum_area))
# total_lung_area = sum(filter(None, lung_area))
# print('ratio: ', total_pneum_area / total_lung_area)
# cv2.waitKey()
path = root_path + '14.dcm'
img = data.load_dcm(path)
flag, pneum_s, lung_s = seg_pneum.seg_pneum(img)
ratio = 0
if lung_s != None:
    ratio  = pneum_s / lung_s
print("lung_area: ", lung_s, "pneum_s", pneum_s, "ratio: ", ratio)
cv2.waitKey()