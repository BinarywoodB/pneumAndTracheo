# -*- coding: utf-8 -*-
# Author = ZRHonor

# import os
import cv2
import data
import seg_pneum
import sys
import os

MODE_TEST = 1
MODE_ALL = 0
mode = MODE_TEST

root_path = '../data/6/'
print("root_path: ", root_path)
if mode == MODE_ALL:
    # folder mode
    pneum_areas = []
    lung_areas = []
    files = os.listdir(root_path)
    files.sort()
    for file in files:
        print(file)
        fileName = root_path + file
        img = data.load_dcm(fileName)
        flag, pneum_area, lung_area = seg_pneum.seg_pneum(img)
        pneum_areas.append(pneum_area)
        lung_areas.append(lung_area)
        print("flag: ", flag, "\tpneum area: ", pneum_area, "\tlung_area: ", lung_area)
        cv2.waitKey(100)

    total_pneum_area = sum(filter(None, pneum_areas))
    total_lung_area = sum(filter(None, lung_areas))
    print("================================")
    print('ratio: ', total_pneum_area / total_lung_area)
else:
    # single test
    fileName = root_path + "14.dcm"
    img = data.load_dcm(fileName)    
    flag, pneum_area, lung_area = seg_pneum.seg_pneum(img)
    print("flag: ", flag, "\tpneum area: ", pneum_area, "\tlung_area: ", lung_area)
    cv2.waitKey()
    print('ratio: ', pneum_area / lung_area)
    
