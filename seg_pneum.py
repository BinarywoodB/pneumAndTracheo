#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZRHonor
"""

import cv2
import numpy as np


def _is_pneum(contour, shape):
    # TODO(ZRHonor): 气胸判断

    # 面积
    area = cv2.contourArea(contour)

    # 重心
    # M = cv2.moments(contour)
    # print(M)
    # cx=int(M['m10']/M['m00'])
    # cy=int(M['m01']/M['m00'])
    
    # if (area<100 or (abs(cx-shape[0])<100 and abs(cy-shape[1])<100)):
    #     return False, 0
    # else:
    #     return True, area
    if area<100:
        return False, 0
    else:
        return True, area


def segmenting(srcimg):
    # TODO(ZRHonor): 图像提取优化, 闭操作等

    # 对图像进行阈值分割，得到胸腔轮廓
    ret, img = cv2.threshold(srcimg, -500, 3071, cv2.THRESH_BINARY)
    img = np.uint8(img)

    # 提取分割结果中的轮廓，并填充孔洞，得到胸腔剖面
    img2, contours, _ = cv2.findContours(
        img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(img.shape, np.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)

    
    # 胸腔剖面去除轮廓，得到肺部剖面
    lung = np.zeros_like(img)
    lung[mask > 127] = 255
    lung[img > 127] = 0
    cv2.imshow('lung',lung)

    # 肺部剖面中黑色的部分就是气胸
    ret, black = cv2.threshold(srcimg, -920, 3071, cv2.THRESH_BINARY)
    pneum = np.zeros_like(img)
    pneum[black < 127] = 255
    pneum[lung < 127] = 0

    # cv2.imshow('result',pneum)
    # cv2.waitKey(5)

    return lung, pneum


def seg_pneum(srcimg):
    lung, pneum = segmenting(srcimg)
    img2, contours, _ = cv2.findContours(
        pneum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 把图像分开，分别判断每个部分是否是气胸
    cv2.imshow('before',pneum)
    pneum_s = 0
    pneum_contours = []
    for contour in contours:
        flag, area = _is_pneum(contour, srcimg.shape)
        if flag:
            pneum_s += area
            pneum_contours.append(contour)
        else:
            cv2.fillPoly(pneum, [contour], 0)
    cv2.imshow('after',pneum)

    if pneum_s == 0:
        return False, None, None, None
    else:
        img2, contours, _ = cv2.findContours(
            lung, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lung_s = 0
        for contour in contours:
            lung_s += cv2.contourArea(contour)

        return True, pneum, pneum_contours, pneum_s / lung_s
