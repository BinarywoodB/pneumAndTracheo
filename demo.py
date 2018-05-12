# -*- coding: utf-8 -*-
# Author = ZRHonor

# import os
import cv2
import data
import seg_pneum
root_path = '5/'
cv2.waitKey()
for i in range(1, 70):
    print(i)
    path = root_path + '{0}.dcm'.format(i)
    img = data.load_dcm(path)
    flag, pneum, pneum_contours, ratio = seg_pneum.seg_pneum(img)
    print(flag, ratio)
    cv2.waitKey(100)

cv2.waitKey()
