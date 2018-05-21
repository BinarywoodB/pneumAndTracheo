#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dlin
"""

import cv2
import numpy as np
import cmath
import math
import copy

import seg_pneum
import situation

MAIN_AREA_LIMIT = 10000  # 用来判断是不是喉部时用到的常数，超过这个值则为CT图中的主体部分
UPPER_LIMIT = -220
LOWER_LIMIT = -1000

# enum status
STATUS_LARYNX = 0        # 咽喉部分。在此之前不进行检测
STATUS_TRACHEA = 1       # 气管阶段。在咽喉部以下，分出支气管之前。特征：圆形度较高，trachea图像只有一个圆孔
STATUS_SWITCH = 2        # 气管转支气管阶段。特征：trachea图像只有一个区域，像两个粘在一起的球。
STATUS_BRONCHUS = 3      # 支气管阶段。特征：trachea图像有两个有一定圆形度的部分。此后不再检测
status = STATUS_LARYNX   # 初始化status为阶段1


# @function detect_stenosis: 处理气管病变的主函数。传入单张CT片，返回是否有狭窄现象，狭窄程度
# @param    src_img:            传入的dcm文件
# @param    last_situation:     上一帧气管情况
# @return   flag:               (true/false)是否患有气管病变（肿瘤 炎症等
# @return   current_situation:  本帧气管情况
def detect_stenosis(src_img, last_situation):
    # resize img to 512*512, 方便显示
    src_img = cv2.resize(src_img, (int(src_img.shape[0] / 2), int(src_img.shape[1] / 2)))
    
    global status
    if status >= STATUS_SWITCH:
        # print("=========== STATUS_SWITCH region. Stop detecting. ==========")
        return False, 0, last_situation
    else:    
        # Step 1: 先判断是否是喉部，喉部不检测（此处气管即使非圆形也为正常现象，不是狭窄）
        img = copy.deepcopy(src_img)   
        flag, main_part_contour = _is_larynx(img)
        if True == flag and STATUS_LARYNX == status:
            # 喉部不检测
            # print("=========== Larynx region. Skip dectection... ==========")
            return False, 0, last_situation
        else:
            # 非喉部，为气管支气管部（或肺部）
            if STATUS_LARYNX == status:
                status = STATUS_TRACHEA
                print("=========== STATUS_TRACHEA region ==========")
            current_situation = get_trachea_region(src_img, main_part_contour, last_situation)
            # 如果发生狭窄
            if current_situation.stenosis_info.stenosis_flag and current_situation.stenosis_info.restore_trachea_area != 0:
                    stenosis_degree = 1 - current_situation.area / current_situation.stenosis_info.restore_trachea_area
            # 没有发生狭窄
            else:
                stenosis_degree = 1

            print("[debug] stenosis area:  ", current_situation.area, \
            "  ; restore_stenosis_area:  ", current_situation.stenosis_info.restore_trachea_area,\
            " ; degree: ", stenosis_degree)

            return True, stenosis_degree, current_situation


# @function get_trachea_region: 得到 气道区域面积 与 还原的原气道区域的面积
#           方法：原图取主体部分，截断后range(-1023, 3167.5)->(-1020, -200)，以阈值-880进行阈值分割，得到气道部分
# @param    src_img:                传入的dcm文件
# @param    main_part_contour:      主体区域轮廓
# @param    last_situation:         上一帧气管情况
# @return   current_situation:      本帧气管情况
def get_trachea_region(src_img, main_part_contour, last_situation): 
    # Step 1. 预处理
    # 1.1 取主体部分
    main_part_img = np.zeros(src_img.shape, np.uint8)
    for contour in main_part_contour:
        cv2.fillPoly(main_part_img, [contour], 255)
    # cv2.imshow('main_part_img', main_part_img)        
    # 1.2 去掉衣服部分 thickness = 20 
    cv2.drawContours(src_img, main_part_contour, -1, (0,0,255), 20)   
    # 1.3 截断
    src_img[main_part_img < 127] = LOWER_LIMIT
    src_img[src_img > UPPER_LIMIT] = UPPER_LIMIT
    src_img[src_img < LOWER_LIMIT] = LOWER_LIMIT
    img = copy.deepcopy(src_img)
    img = np.uint8(img)
    # cv2.imshow("src_img", img)

    # Step 2. 拿到内部腔体（气道和肺部），形状较精确
    # 2.1 src_img阈值分割以肺CT值切分，即内部腔体和其他组织，保留 软组织部分形成的胸腔轮廓 部分
    ret, soft_tissue = cv2.threshold(src_img, -600, 3071, cv2.THRESH_BINARY)
    soft_tissue = np.uint8(soft_tissue)
    # cv2.imshow('soft_tissue', soft_tissue)
    # 2.2 由胸腔轮廓与整个胸腔区域对比得到内部的腔体（包含气道和肺部）
    cavity = np.zeros_like(src_img)
    cavity[main_part_img > 127] = 255
    cavity[soft_tissue > 127] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(cavity, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('cavity_after_opening', opening)
    # 2.3 把肺叶中的肺泡等空隙填充起来
    opening = np.uint8(opening)
    img2, contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(opening, [contour], 255)  
    cv2.imshow("opening fill ", opening)

    # Step 3. 拿到空气CT值部分
    # 3.1 src_img阈值分割以空气值切分，即接近空气CT值部分和其他部分（肺叶|组织）
    ret, air_part = cv2.threshold(src_img, -860, 3071, cv2.THRESH_BINARY)
    # cv2.imshow('air_part', air_part)
    # 3.2 由气道部分和整个胸腔主体区域对比得到 空气CT值部分（有气道也有肺泡干扰）
    main_air_part = np.zeros_like(src_img, np.uint8)
    main_air_part[main_part_img > 127] = 255
    main_air_part[air_part > 127] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    main_air_part = cv2.morphologyEx(main_air_part, cv2.MORPH_OPEN, kernel)
    cv2.imshow('main_air_part', main_air_part)

    # Step 4. 检查Step 2拿到的内部腔体共有几大块。对每个部分（有可能是气道也有可能是肺叶），筛选出气道部分
    img2, contours, _ = cv2.findContours(
        opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    trachea_contours = []
    current_situation = copy.deepcopy(last_situation)
    for contour in contours:
        # 4.1 面积筛选。气管支气管的面积在1000以内，气管大约在240～550，支气管大约在150～200逐渐变小
        area = cv2.contourArea(contour)
        if area > 1000:  
            continue
        # 4.2 在main_air_part中找到opening的每一块对应的部分，
        #     面积为:   blur_area -> main_air_part
        #              area      -> opening（preciser）
        #     若 0.33area < blur_area  则判定为气管区域。计算圆形度等
        tmp_img = np.zeros_like(src_img, np.uint8)
        cv2.fillPoly(tmp_img, [contour], 255)
        blur_area = sum(main_air_part[tmp_img > 127]) / 255
        if area / 3 < blur_area:
            trachea_contours.append(contour)
    print("trachea size: ", len(trachea_contours))

    
    fake_trachea = np.zeros_like(src_img, np.uint8)
    for contour in trachea_contours:
        cv2.fillPoly(fake_trachea, [contour], 255)  
    if trachea_contours != []:
        cv2.imshow('fake_trachea', fake_trachea)
    
    global status
    # Step 5: 状态机判断
    if status == STATUS_TRACHEA:
        # status是气管阶段    
        if len(trachea_contours) > 1:
            # 在status为气管阶段时气道数量超过一个，有可能是食道或肺泡带来的干扰（不会是支气管），借助上一帧气管所在位置判定保留气管，删除干扰
            # 两帧重合度大于0.8的话判定为同一孔洞
            trachea_contours = rule_out_esophagus(trachea_contours, src_img, last_situation.contour)
            print("new trachea contour size: ", len(trachea_contours))
        
        # 去除食道、肺泡等的干扰后，若只有一个气孔，则判断是否有狭窄和是否进入转换状态
        if 1 == len(trachea_contours):
            current_situation.contour = trachea_contours[0]
            [current_situation.area, _, current_situation.perimeter, current_situation.circularity] = calculate_circularity(current_situation.contour)
            if [] != last_situation.contour:
                current_situation.stenosis_info.restore_trachea_area = current_situation.area
                
                # 狭窄状态中
                if last_situation.stenosis_info.stenosis_flag:
                    current_situation.stenosis_info.restore_trachea_area = last_situation.stenosis_info.restore_trachea_area            
                    # 狭窄状态中，判断是否恢复：面积变大，圆形度高于0.87，出现连续两帧可从狭窄状态中恢复
                    if current_situation.circularity > 0.87 and current_situation.area > last_situation.area:
                        print("although still stenosis, we believe we are recovering...")
                        current_situation.stenosis_info.recover_count  = last_situation.stenosis_info.recover_count + 1
                        if 2 == last_situation.stenosis_info.recover_count:
                            print("#### 确认从狭窄状态中复原。we now recover from stenosis.")
                            current_situation.stenosis_info.restore_trachea_area = current_situation.area
                            current_situation.stenosis_info.stenosis_flag = False
                            current_situation.stenosis_end_area = current_situation.area
                            print("#### get end area: ", current_situation.stenosis_end_area)
                else:
                    # 非狭窄状态中，判断是否狭窄：面积变小，圆形度低于0.87。这种情况超过连续两帧判定为狭窄
                    if current_situation.circularity < 0.87 and current_situation.area < last_situation.area:
                        print("likely we reach stenosis point...")
                        current_situation.stenosis_info.stenosis_count  = last_situation.stenosis_info.stenosis_count + 1
                        if 2 == last_situation.stenosis_info.stenosis_count:
                            current_situation.stenosis_info.restore_trachea_area = last_situation.stenosis_info.restore_trachea_area
                            print("### 确认狭窄。还原区域面积： ", current_situation.stenosis_info.restore_trachea_area)
                            current_situation.stenosis_info.stenosis_flag = True
                            current_situation.stenosis_start_area = last_situation.stenosis_info.restore_trachea_area
                            print("#### get start area: ", current_situation.stenosis_start_area)
                    else:
                        if last_situation.stenosis_info.stenosis_count == 1:
                            print("### 虚惊一场，并不是狭窄。But it turns out it's only false alarm")
                        current_situation.stenosis_info.stenosis_count = 0

                # 判断是否进入转换状态：面积变大，周长变大，圆形度却下降且圆形度低于0.85。这种情况超过连续两帧判定为进入STATUS_SWITCH阶段
                if (current_situation.area > last_situation.area and 
                current_situation.perimeter > last_situation.perimeter and
                current_situation.circularity < last_situation.circularity and
                current_situation.circularity < 0.85):
                    print("likely we reach switching point...")
                    current_situation.switch_count  = last_situation.switch_count + 1
                    if 2 == current_situation.switch_count:
                        status = STATUS_SWITCH
                        print("WE HAVE REACH STATUS_SWITCH !!!")
                else:
                    current_situation.switch_count = 0
            else:
                print("上一帧气管位置为空是怎么回事？")
    elif STATUS_SWITCH == status:
        print("=========== STATUS_SWITCH region. Stop detecting. ==========")
        
    trachea = np.zeros_like(src_img, np.uint8)
    for contour in trachea_contours:
        cv2.fillPoly(trachea, [contour], 255)  
    if trachea_contours != []:
        cv2.imshow('trachea', trachea)
    
    return current_situation

def rule_out_esophagus(trachea_contours, src_img, last_contour):
    new_trachea_contours = []
    for contour in trachea_contours:
        overlap = 0
        tmp_img = np.zeros_like(src_img, np.uint8)
        cv2.fillPoly(tmp_img, [contour], 255)
        cv2.imshow("tmp_img", tmp_img)        
        last_img = np.zeros_like(src_img, np.uint8)
        cv2.fillPoly(last_img, [last_contour], 255)
        cv2.imshow("last post", last_img)
        overlap = (sum( tmp_img[last_img > 127] ) / 255) / (sum( tmp_img[tmp_img > 127] ) / 255)
        print("overlap: ", overlap)
        if overlap > 0.8:
            new_trachea_contours.append(contour)
    return new_trachea_contours

# @function _is_larynx: 判断传入的图片是否是喉部，并返回主体部分的轮廓
#                       根据大块区域的数量判断是否是喉部，大块区域数量大于等于3 或 只有一个大块区域且面积小于4000，则为喉部
# @param    src_img:            传入的dcm文件
# @return   flag:               (true/false)是否是喉部
# @return   main_part_contour:  保留的主体区域的轮廓
def _is_larynx(src_img):
    # Step 0: 对图像进行阈值分割，二值化为两部分：（气管支气管气胸，其他组织）
    # 阈值-450：只需大致划分，便于后续拿到主要部分
    thres_get_main_part = -450
    ret, img = cv2.threshold(src_img, thres_get_main_part, 3071, cv2.THRESH_BINARY)
    # cv2.imshow('binary img: -450', img)    
    img = np.uint8(img)
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("[debug] contour size: ", len(contours))
    main_part_contour = []
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        # print("[debug] area: ", area)
        if area > MAIN_AREA_LIMIT:
            main_part_contour.append(contour)
            # print("[debug] contour(added to main_part_contour) 's area: ", area)
            if area > max_area:
                max_area = area
    # print("[debug] main_part_contour size: ", len(main_part_contour))
    if len(main_part_contour) >= 3 or (1 == len(main_part_contour) and max_area < 4000):
        return True, main_part_contour
    else:
        return False, main_part_contour    

def get_max_contour_area(src_img):
    img = copy.deepcopy(src_img)
    img = np.uint8(img)
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        max_area = max(max_area, cv2.contourArea(contour))
    return max_area

def calculate_circularity(contour):    
        # 计算圆形度
        area = cv2.contourArea(contour)
        radius = (area / math.pi) ** 0.5
        equivDiameter = 2 * radius
        perimeter = cv2.arcLength(contour, True)
        circularity = math.pi * equivDiameter / perimeter
        # print("面积： ", area, " 半径： ", radius, " 周长： ", perimeter, " 圆形度： ", circularity)
        print("面积： ", area, " 圆形度： ", circularity)
        return area, radius, perimeter, circularity