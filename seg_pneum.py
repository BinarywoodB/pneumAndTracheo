#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZRHonor
"""

import cv2
import numpy as np
import cmath
import math
import copy

MAIN_AREA_LIMIT = 10000  # 用来判断是不是喉部时用到的常数，超过这个值则为CT图中的主体部分
UPPER_LIMIT = -220
LOWER_LIMIT = -1000

# @function seg_pneum: 主函数：气胸分割算法的入口
# @param    src_img:                传入的dcm文件（单张CT图像）
# @return   flag:                   此张CT图像是否具有气胸现象（True/False）
# @return   pneum_area:             此张CT图像中气胸面积
# @return   lung_area:              此张CT图像中胸腔面积
def seg_pneum(src_img):
    # cv2.imshow('origin image:', src_img)
    # Step 1: 分割得到 胸腔区域lung 和 气胸区域pneum
    lung, pneum = segmenting(src_img)

    # Step 2: 计算胸腔面积和气胸面积
    pneum_area = 0
    img2, contours, _ = cv2.findContours(
        pneum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:       
        pneum_area += cv2.contourArea(contour) 
    lung_area = 0
    img2, contours, _ = cv2.findContours(
        lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:       
        lung_area += cv2.contourArea(contour) 
    
    flag = False
    if pneum_area != 0:
        flag = True
    
    return flag, pneum_area, lung_area


# @function segmenting: 分割得到 胸腔区域lung 和 气胸区域pneum
# @param    src_img:                传入的dcm文件（单张CT图像）
# @return   lung:                   胸腔区域
# @return   pneum:                  气胸区域
def segmenting(src_img):
    # Step 1: 预处理
    src_img, main_part_img = preprocess_src_img(src_img)

    # Step 2: 拿到内部腔体（气道和肺部)
    # 2.1 对图像进行阈值分割，得到胸腔轮廓
    lung_hu = -600 # lung HU is -700 to −600
    ret, soft_tissue = cv2.threshold(src_img, lung_hu, 3071, cv2.THRESH_BINARY)
    soft_tissue = np.uint8(soft_tissue)
    # cv2.imshow('soft_tissue', soft_tissue)
    # 2.2 由胸腔轮廓与整个胸腔区域对比得到内部的腔体（包含气道和肺部）
    cavity = np.zeros(src_img.shape, np.uint8)
    cavity[main_part_img > 127] = 255
    cavity[soft_tissue > 127] = 0
    # cv2.imshow('cavity', cavity)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cavity = cv2.morphologyEx(cavity, cv2.MORPH_OPEN, kernel)
    # 2.3 把肺叶中的肺泡等空隙填充起来
    img2, contours, _ = cv2.findContours(
        cavity, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.fillPoly(cavity, [contour], 255)  
    cv2.imshow('cavity fill', cavity)
    
    # Step 3: 去除气管支气管、食道等区域的干扰
    lung = _rule_out_tracheal_area(cavity)
    cv2.imshow("lung", lung)
    
    src_img[lung < 127] = src_img.min()
    # air HU: -1000
    ret, black = cv2.threshold(src_img, -920, 3071, cv2.THRESH_BINARY)
    pneum = np.zeros(src_img.shape, np.uint8)    
    pneum[black < 127] = 255
    pneum[lung < 127] = 0
    cv2.imshow('black', black)   
    cv2.imshow('pneum', pneum)    

    return lung, pneum

# @function preprocess_src_img:     预处理图像，取主体部分，去掉衣服部分，并截断Hu值
# @param    src_img:                原始图像
# @return   src_img:                预处理过后的图像：Hu值被截断，保留主体部分并去除外围衣服
# @return   main_part_img:          uint8格式的主体部分图像，range(0, 255)
def preprocess_src_img(src_img):
    # Step 1: 取主体部分
    # 对图像进行阈值分割，二值化为两部分：（气管支气管气胸，其他组织）
    # 阈值-450：只需大致划分，便于后续拿到主要部分
    thres_get_main_part = -450
    ret, img = cv2.threshold(src_img, thres_get_main_part, 3071, cv2.THRESH_BINARY)
    img = np.uint8(img)
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_part_contour = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MAIN_AREA_LIMIT:
            main_part_contour.append(contour)
    main_part_img = np.zeros(src_img.shape, np.uint8)
    for contour in main_part_contour:
        cv2.fillPoly(main_part_img, [contour], 255)

    # Step 2: 去掉衣服部分 thickness = 20 
    cv2.drawContours(src_img, main_part_contour, -1, (0,0,255), 20)   
    # Step 3: 截断
    src_img[main_part_img < 127] = LOWER_LIMIT
    src_img[src_img > UPPER_LIMIT] = UPPER_LIMIT
    src_img[src_img < LOWER_LIMIT] = LOWER_LIMIT

    return src_img, main_part_img


def _rule_out_tracheal_area(src_img):
    # 气胸判断.用于去除气管支气管区域
    # 气管支气管区域判断依据：面积大于20且圆形度大于0.8
    img2, contours, _ = cv2.findContours(
        src_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # 计算圆形度
        area = cv2.contourArea(contour)
        if area == 0:
            continue      
        radius = (area / math.pi) ** 0.5
        equivDiameter = 2 * radius
        perimeter = cv2.arcLength(contour, True)
        circularity = math.pi * equivDiameter / perimeter
        # print("面积： ", area, " 半径： ", radius, " 周长： ", perimeter, " 圆形度： ", circularity)
        if (area > 20) and (area < 500) and (circularity > 0.8):
            # print("qiguan...")
            cv2.fillPoly(src_img, [contour], 0)
    return src_img

def refinePneum(pneum):    
    # 精细化处理气胸，先筛除气管支气管区域，再通过闭操作平滑图像，得到真实气胸区域
    # cv2.imshow('[pneum] before partition', pneum)

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

