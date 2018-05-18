#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZRHonor
"""

import cv2
import numpy as np
import cmath
import math

def _rule_out_tracheal_area(pneum):
    # 气胸判断.用于去除气管支气管区域
    # 气管支气管区域判断依据：面积大于20且圆形度大于0.8
    img2, contours, _ = cv2.findContours(
        pneum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 计算圆形度
        area = cv2.contourArea(contour)
        if area == 0:
            continue      
        radius = (area / math.pi) ** 0.5
        equivDiameter = 2 * radius
        perimeter = cv2.arcLength(contour, True)
        circularity = math.pi * equivDiameter / perimeter
        print("面积： ", area, " 半径： ", radius, " 周长： ", perimeter, " 圆形度： ", circularity)
        if (area > 20) and (circularity > 0.8):
            print("qiguan...")
            cv2.fillPoly(pneum, [contour], 0)
    return pneum

# 分割得到 胸廓lung 和 气胸轮廓pneum
def segmenting(srcimg):
    # TODO(ZRHonor): 图像提取优化, 闭操作等

    # 对图像进行阈值分割，得到胸腔轮廓
    lung_hu = -600 # lung HU is -700 to −600
    ret, img = cv2.threshold(srcimg, lung_hu, 3071, cv2.THRESH_BINARY)
    img = np.uint8(img)
    cv2.imshow('chest', img)

    # 提取分割结果中的轮廓，并填充孔洞，得到胸腔剖面
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)    
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    cv2.imshow('mask', mask)

    # 找到胸腔剖面的重心，如果出现气管支气管应该在这附近
    
    # 胸腔剖面去除轮廓，得到肺部剖面
    lung = np.zeros_like(img)
    lung[mask > 127] = 255
    lung[img > 127] = 0
    cv2.imshow('lung',lung)
    # TODO: lung 开操作后仅保留两个肺腔，再闭操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(lung, cv2.MORPH_OPEN, kernel)
    img2, contours, _ = cv2.findContours(
        lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find first 2 largest area
    lung_contours = []
    max_area = 0
    second_max_area = 0
    max_area_contour = []
    second_max_area_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # print("area: ", area)
        if area > max_area:
            second_max_area = max_area
            max_area = area
            second_max_area_contour = max_area_contour
            max_area_contour = contour
    # print("max_area: ", max_area, "; second max area: ", second_max_area)
    if max_area == 0:
        return None, None
    elif second_max_area == 0:
        lung_contours.append(max_area_contour)
    else:
        lung_contours.append(max_area_contour)
        lung_contours.append(second_max_area_contour)

    # draw lung    
    mask = np.zeros(img.shape, np.uint8)    
    for contour in lung_contours:
        cv2.fillPoly(mask, [contour], 255)
    cv2.imshow('real lung', mask)

    # 肺部剖面中黑色的部分就是气胸
    # air HU: -1000
    ret, black = cv2.threshold(srcimg, -920, 3071, cv2.THRESH_BINARY)
    pneum = np.zeros_like(img)
    pneum[black < 127] = 255
    pneum[lung < 127] = 0
    cv2.imshow('black', black)
    # cv2.imshow('lung', lung)    
    cv2.imshow('pneum00', pneum)    

    # cv2.imshow('result',pneum)
    # cv2.waitKey(5)

    return lung, pneum

def refinePneum(pneum):    
    # 精细化处理气胸，先筛除气管支气管区域，再通过闭操作平滑图像，得到真实气胸区域
    cv2.imshow('[pneum] before partition', pneum)

    # 先按圆形度去除气管支气管区域
    pneum_area = 0
    pneum_contours = []
    pneum = _rule_out_tracheal_area(pneum)
    cv2.imshow('_rule_out_tracheal_area', pneum)
    
    # 膨胀腐蚀操作(闭操作)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    dilation = cv2.dilate(pneum, kernel, iterations = 1)
    pneum = cv2.erode(dilation, kernel, iterations = 1)
    cv2.imshow('close operation', pneum)

    return pneum

def seg_pneum(src_img):
    # 主函数：气胸分割算法的入口
    # 得到胸廓面积和气胸区域面积
    cv2.imshow('origin image:', src_img)
    # Step 0: 分割得到 胸腔区域lung 和 疑似气胸区域pneum
    lung, pneum = segmenting(src_img)
    # Step 1: 精细化气胸区域
    pneum = refinePneum(pneum)

    pneum_area = 0
    img2, contours, _ = cv2.findContours(
        pneum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:       
        pneum_area += cv2.contourArea(contour) 
        
    if pneum_area == 0:
        return False, None, None
    else:
        img2, contours, _ = cv2.findContours(
            lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lung_s = 0	
        cv2.drawContours(src_img, contours, -1, (0, 0, 255), 2)
        for contour in contours:
            lung_s += cv2.contourArea(contour)

        return True, pneum_area, lung_s
