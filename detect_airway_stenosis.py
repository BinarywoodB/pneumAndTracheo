# -*- coding: utf-8 -*-
# Author: dlin
"""
    detect_airway_stenosis.py detects tracheobronchial stenosis,
    giving out :
        1. flag:                    该病患是否出现气管支气管狭窄现象
        2. max_degree:              出现气道狭窄最严重的地方的狭窄程度（1 - 该处截面面积 / 发生狭窄前后气管截面面积的均值）-- 1:完全堵塞 | 0:没有狭窄
        3. min_sectional_area:      出现气道狭窄最严重的地方的面积
        4. min_sectional_area_file: 出现气道狭窄最严重的地方的文件名
        2. lesions:                 单张CT片的情况。包括文件名，气道截面面积，参考狭窄程度
           lesions = [["00000016.dcm", 0.75], ["00000017.dcm", 0.72], ["00000018.dcm", 0.73]]
        3. maximum degree of stenosis of the dcm folder
"""
import os
import cv2
import sys
import copy

import situation
import data
import detect_stenosis

if len(sys.argv) < 2:
    print("please pass in mode: 1 for test | 0 for whole folder")
    exit()
single_test = int(sys.argv[1])
print("single_test: ", single_test)

if single_test == 1:
    root_path = '/Users/lindi/workspace/graduationDesign/qiguan/1/'
    # for test: 单张测试
    print("root_path: ", root_path)
    test_path = root_path + '00000125.dcm'
    # corner_case: 00000160.dcm -- 
    img = data.load_dcm(test_path)
    last_situation = situation.Situation()
    flag, degree, current_situation = detect_stenosis.detect_stenosis(img, last_situation)    
    print("flag: ", flag)
    # print("degree: ", degree)
    cv2.waitKey()
    cv2.destroyAllWindows()
elif single_test == 0:
    # 遍历某个患者的dcm folder中的每一张CT slice
    root_path = '/Users/lindi/workspace/graduationDesign/qiguan/1/'
    print("root_path: ", root_path)
    lesions = []
    max_degree = 0
    min_sectional_area = 1000
    min_sectional_area_file = ""
    last_situation = situation.Situation()
    files = os.listdir(root_path)
    files.sort()
    for file in files:
        print(file)
        fileName = root_path + file
        img = data.load_dcm(fileName)
        flag, degree, current_situation = detect_stenosis.detect_stenosis(img, last_situation)    
        last_situation = copy.deepcopy(current_situation)

        if flag == True:
            if current_situation.area < min_sectional_area:
                min_sectional_area_file = file
                min_sectional_area = current_situation.area

            lesion = [file, degree]
            lesions.append(lesion)
            max_degree = min(max_degree, degree)
        cv2.waitKey(5)
        if file == '00000125.dcm':
            cv2.waitKey()
    print("max_degree: ", max_degree)

    if current_situation.stenosis_start_area != -1 and current_situation.stenosis_end_area != -1:
        airway_area = (current_situation.stenosis_start_area + current_situation.stenosis_end_area) / 2
        if airway_area != 0:
            max_degree = 1 - min_sectional_area / airway_area
            print("min_sectional_area: ", min_sectional_area, " start: ", current_situation.stenosis_start_area, " end: ", current_situation.stenosis_end_area)
    print("============================")
    if 0 != max_degree:
        print("出现气管支气管狭窄!!!")
        print("出现气道狭窄最严重的地方的文件名:\t", min_sectional_area_file)
        print("出现气道狭窄最严重的地方的狭窄程度:\t", max_degree)
        print("出现气道狭窄最严重的地方的面积:\t\t", min_sectional_area)
        print("lesions: ", lesions)
        
    else:
        print("没有狭窄发生 :)")