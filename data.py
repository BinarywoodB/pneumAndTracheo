# -*- coding: utf-8 -*-
# Author = ZRHonor

import cv2
import numpy
import dicom
# import gdcm
from matplotlib import pyplot as plt


def load_dcm(path):
    dcm = dicom.read_file(path)
    dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    # 获取图像中的像素数据
    slices = []
    slices.append(dcm)

    # 复制Dicom图像中的像素数据
    srcimg = slices[int(len(slices) / 2)].image.copy()
    return srcimg


def main(path):
    srcimg = load_dcm(path)
    cv2.imshow('src', srcimg)
    cv2.waitKey(5)

    # 对图像进行阈值分割
    ret, img = cv2.threshold(srcimg, -500, 3071, cv2.THRESH_BINARY)
    img = numpy.uint8(img)
    # cv2.imshow('debug1',img)
    # cv2.waitKey(5)

    # 提取分割结果中的轮廓，并填充孔洞
    img2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = numpy.zeros(img.shape, numpy.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    # img[(mask > 0)] = 255

    lungs = numpy.zeros_like(img)
    lungs[mask > 0] = 255
    # cv2.imshow('ceshi', lungs)
    # cv2.waitKey(5)
    lungs[img == 255] = 0
    # cv2.imshow('lungs', lungs)
    # cv2.waitKey(5)
    # mask = temp
    # img[(temp > 0)] = 255
    img = lungs
    img2, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = numpy.zeros(img.shape, numpy.uint8)
    for contour in contours:
        cv2.fillPoly(mask, [contour], 255)
    img[(mask > 0)] = 255
    # cv2.imshow('lungs2', img)
    # cv2.waitKey(5)
    ret, object = cv2.threshold(srcimg, -880, 3071, cv2.THRESH_BINARY)
    pneum = numpy.zeros_like(object)
    pneum[object < 127] = 255
    pneum[lungs < 127] = 0
    img = pneum
    cv2.imshow('pneum', pneum)
    cv2.waitKey(5)
    # 对分割结果进行形态学的开操作
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('debug3', img)
    # cv2.waitKey(5)
    # 根据分割mask获取分割结果的像素数据
    # img2 = slices[int(len(slices) / 2)].image.copy()
    # img2[(img == 0)] = -2000
    # cv2.imshow('debug4', img)
    # cv2.waitKey(5)
    # 显式原始数据，mask和分割结果
    # plt.figure(figsize=(24, 24))
    # plt.subplot(131)
    # plt.imshow(slices[int(len(slices) / 2)].image, 'gray')
    # plt.title('Original')
    # # plt.subplot(132)
    # # plt.imshow(img, 'gray')
    # # plt.title('Mask')
    # plt.subplot(133)
    # plt.imshow(img2, 'gray')
    # plt.title('Result')
    # plt.show()
    # plt.close()