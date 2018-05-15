#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: dlin
"""

import cv2
import numpy as np
import cmath
import math

MAIN_AREA_LIMIT = 10000  # 用来判断是不是喉部时用到的常数，超过这个值则为CT图中的主体部分

# @function detect_stenosis: 处理气管病变的主函数。传入单张CT片，返回是否有狭窄现象，狭窄程度
# @param    src_img:    传入的dcm文件
# @return   flag:       (true/false)是否患有气管病变（肿瘤 炎症等）
# @return   degree:     degree of stenosis: decrease in cross-sectional area
def detect_stenosis(src_img):
    # resize img to 512*512
    img = cv2.resize(src_img, (int(src_img.shape[0] / 2), int(src_img.shape[1] / 2)))
    cv2.imshow('origin', img)
    flag = True
    degree = 0.3

    # 处理图像的主函数
    flag, degree = main_process(img)
    return flag, degree

def main_process(src_img):
    # Step 0: 预处理：对图像进行阈值分割，二值化
    # 只需大致划分，便于后续拿到主轮廓
    ret, img = cv2.threshold(src_img, -450, 3071, cv2.THRESH_BINARY)
    img = np.uint8(img)
    # cv2.imshow('binary img', img)

    # Step 1: 先判断是否是喉部，喉部不检测（此处即使非圆形也为正常现象，不是狭窄）
    # 开操作，根据圆孔所在的区域的面积判断是否是喉部
    flag, main_part_contour = _is_larynx(img)
    if True == flag:
        # 喉部不检测
        print("=========== Larynx region. Skip dectection... ==========")
        return False, 0
    else:    
        # 非喉部，为气管支气管部（或肺部）。
        print("=========== tracheobronchial region ==========")        
        # Step 2: 得到主要区域 main_part_img。即去除肩部、CT扫描仪底部等后的颈部或胸部区域
        main_part_img = np.zeros(src_img.shape, np.uint8)
        for contour in main_part_contour:
            cv2.fillPoly(main_part_img, [contour], 255)
        cv2.imshow('main_part_img', main_part_img)

        # Step 3: 得到main_air_part_img -- 主体区域中CT值接近空气的部分，即可能包含气管支气管与肺部气胸
        # 3.1 对图像进行阈值分割，二值化，拿到全图中和空气CT值接近的部分 air_part_img
        ret, air_part_img = cv2.threshold(src_img, -900, 3071, cv2.THRESH_BINARY)
        air_part_img = np.uint8(air_part_img)
        cv2.imshow('air_part_img', air_part_img)
        main_air_part_img = np.zeros(src_img.shape, np.uint8)
        # 3.2 主要区域main_part_img 与 全图空气CT值接近的部分air_part_img 相比较后得到 主要区域中与空气CT值接近的部分，即main_air_part_img
        main_air_part_img[main_part_img > 127] = 255
        main_air_part_img[air_part_img > 127] = 0
        img2, contours, _ = cv2.findContours(
            main_air_part_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.fillPoly(main_air_part_img, [contour], 255)
        cv2.imshow('main_air_part_img', main_air_part_img)
        
        # 3.3 去掉肺部中细小的空气得到气管或支气管
        # 保留最大的区域
        # TODO：此处不考虑有气胸的情况。应有更好的判断方法
        tracheo = np.zeros(src_img.shape, np.uint8)
        max_tracheo_area = 0
        tracheo_contour = []
        img2, contours, _ = cv2.findContours(
            main_air_part_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            print("[debug] tracheo area: ", area)
            if area > max_tracheo_area and area > 1.0:
                max_tracheo_area = area
                tracheo_contour = contour
        if tracheo_contour != []:
            cv2.fillPoly(tracheo, [tracheo_contour], 255)
        cv2.imshow('tracheo', tracheo)
    
        # 计算气管支气管的横截面积下降程度
        degree = 0
        return True, degree

# @function _is_larynx: 判断传入的图片是否是喉部
# 开操作，根据圆孔所在的区域的面积大小判断是否是喉部，若非喉部，则返回主体区域的轮廓
# @param    src_img:            传入的dcm文件
# @return   flag:               (true/false)是否是喉部
# @return   main_part_contour:  保留的主体区域的轮廓
def _is_larynx(src_img):
    img = open_operation(src_img, kernel_size = 1)    
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("[debug] contour size: ", len(contours))
    main_part_contour = []
    max_area = 0
    max_area_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        print("[debug] area: ", area)
        if area > MAIN_AREA_LIMIT:
            main_part_contour.append(contour)
            print("[debug] contour(added to main_part_contour) 's area: ", area)
            if area > max_area:
                max_area = area
                max_area_contour = contour
    print("[debug] main_part_contour size: ", len(main_part_contour))
    if len(main_part_contour) >= 3 or (1 == len(main_part_contour) and max_area < 4000):
        return True, main_part_contour
    else:
        return False, main_part_contour    

# 开操作：先腐蚀再膨胀
def open_operation(src_img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))   
    erosion = cv2.erode(src_img, kernel, iterations = 1)
    # cv2.imshow('[debug] lung erosion', erosion)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    # cv2.imshow('[debug] lung dilation', dilation)
    return dilation
    
# 闭操作：先膨胀再腐蚀
def close_operation(src_img, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)) 
    dilation = cv2.dilate(src_img, kernel, iterations = 1)
    cv2.imshow('[debug] lung dilation', dilation)  
    erosion = cv2.erode(dilation, kernel, iterations = 1)
    cv2.imshow('[debug] lung erosion', erosion)
    return erosion