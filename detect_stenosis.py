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
# @function handle_qiguan: 处理气管病变的主函数。传入单张CT片，返回是否有狭窄现象，狭窄程度
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
    # 预处理：对图像进行阈值分割，二值化
    lung_hu = -500 # lung HU is -700 to −600
    ret, img = cv2.threshold(src_img, lung_hu, 3071, cv2.THRESH_BINARY)
    img = np.uint8(img)
    cv2.imshow('binary img', img)

    # 先判断是否是喉部，喉部不检测（此处即使非圆形也为正常现象，不是狭窄）
    # 开操作，根据圆孔所在的区域的面积判断是否是喉部
    flag, filtered_contour = _is_larynx(img)
    if True == flag:
        # 喉部不检测
        print("Larynx area. Skip dectection...")
        return False, 0
    else:    
        # 得到主要区域
        filtered_img = np.zeros(img.shape, np.uint8)    
        for contour in filtered_contour:
            cv2.fillPoly(filtered_img, [contour], 255)
        cv2.imshow('filtered_img', filtered_img)

        # 得到主要区域的原二值图
        # body为主体区域中CT值接近空气的部分，即可能包含气管支气管与肺部
        body = np.zeros(img.shape, np.uint8)
        body[filtered_img > 127] = 255
        body[img > 127] = 0
        img2, contours, _ = cv2.findContours(
            body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow('body', body)
        for contour in contours:
            cv2.fillPoly(body, [contour], 255)
        cv2.imshow('body', body)
        
        # air HU: -1000
        ret, black = cv2.threshold(src_img, -900, 3071, cv2.THRESH_BINARY)
        tracheo = np.zeros_like(img)
        cv2.imshow('black', black)
        
        tracheo[body > 127] = 255
        tracheo[black > 127] = 0
        cv2.imshow('tracheo', tracheo)
    
        # 计算气管支气管的横截面积下降程度
        degree = 0
        return True, degree

# @function _is_larynx: 判断传入的图片是否是喉部
# 开操作，根据圆孔所在的区域的面积大小判断是否是喉部，若非喉部，则返回主体区域的轮廓
# @param    src_img:            传入的dcm文件
# @return   flag:               (true/false)是否是喉部
# @return   filtered_contour:   保留的主体区域的轮廓
def _is_larynx(src_img):
    img = open_operation(src_img, kernel_size = 5)    
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print("[debug] contour size: ", len(contours))
    filtered_contour = []
    max_area = 0
    max_area_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MAIN_AREA_LIMIT:
            filtered_contour.append(contour)
            print("[debug] area: ", area)
            if area > max_area:
                max_area = area
                max_area_contour = contour
    # print("[debug] filtered_contour size: ", len(filtered_contour))
    if len(filtered_contour) >= 3 or (1 == len(filtered_contour) and max_area < 40000):
        return True, filtered_contour
    else:
        return False, filtered_contour    

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