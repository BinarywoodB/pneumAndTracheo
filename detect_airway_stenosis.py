# -*- coding: utf-8 -*-
# Author: dlin
"""
    detect_airway_stenosis.py detects tracheobronchial stenosis,
    giving out :
        1. flag indicating whether or not the patient of the given dcm folder has tracheobronchial stenosis
        2. array containing lesion area slices and degree of stenosis
           lesions = [["00000016.dcm", 0.75], ["00000017.dcm", 0.72], ["00000018.dcm", 0.73]]
        3. maximum degree of stenosis of the dcm folder
"""
import os
import cv2
import data
import detect_stenosis
import sys

root_path = 'AirwayStenosis/20160102/NI JU PING_00924781_165829/1.0 x 1.0_20160102_170045/'

# for test: 单张测试
path = root_path + '00000119.dcm'
img = data.load_dcm(path)
flag, degree = detect_stenosis.detect_stenosis(img)
print("flag: ", flag)
print("degree: ", degree)
cv2.waitKey()
# cv2.destroyAllWindows()

# # 遍历某个患者的dcm folder中的每一张CT slice
# lesions = []
# max_degree = 0
# for parent, dirnames, filenames in os.walk(root_path):
#     for filename in filenames:
#         path = os.path.join(parent,filename)
#         print(path)
#         img = data.load_dcm(path)
#         flag, degree = detect_stenosis.detect_stenosis(img)
#         if flag == True:
#             lesion = [filename, degree]
#             lesions.append(lesion)
#             if degree > max_degree:
#                 max_degree = degree
# if max_degree >0:
#     print("============================")
#     print("tracheobronchial stenosis!!!")
#     print("lesions: ", lesions)
#     print("max_degree: ", max_degree)
# else:
#     print("SAFE :)")