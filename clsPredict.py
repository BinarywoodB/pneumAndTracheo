#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:42:59 2017

@author: mjj
"""


import cv2
import os
import numpy as np
import h5py

import SimpleITK as sitk
import pickle
#import tensorflow as tf
#import init_sys

#from flip_regression import flip_predict
from LungSegmentation import lung_segmentation_scan

import faster_rcnn.init as frcn_init
import fp_reduction.init as fp_init
#from faster_rcnn.predict import frcn_predict
import faster_rcnn.tools._init_paths
from fast_rcnn.test import test_net

#import fp_reduction.convert_candidate_file as convert_candidate_file
import fp_reduction.pipeline as pipeline
from fp_reduction.easy_io import H5Writer
from fp_reduction.prepare_h5_file_with_interpolation_with_multiple_h5 import gen

#from predict_cmd import main
#from fp_reduction.predict import predict_model

from fp_reduction.predict_config import get_config

from scipy.ndimage import zoom
from nodule_seg import nodule_seg
import numpy as np

class Predict(object):
    net = None
#    model = None
    thr = 0.01
    
    def __init__(self, dataDir, gpuID=0):
        self.gpu_id = gpuID
        self.paraScale=1
        #self.init_sys()
        
#        self.cacheDir = './cache'
        #self.build_data(dataDir)

    def init_sys(self):  
        self.modelDir = './data'
        #    model = fp_init.init(data_path)
        if self.net is None:
            Predict.net = frcn_init.init(self.gpu_id, self.modelDir)
#        if self.model is None:
#            predict.model = fp_init.init(self.modelDir)

    def build_data(self, indir):
        self.id = os.path.basename(indir)
        self.cacheDir = os.path.join('./cache', self.id)
        
        if not os.path.exists(self.cacheDir):
            os.makedirs(self.cacheDir)
            
        self.vol_file = os.path.join(self.cacheDir,'vol.hdf5')
        
        reader = sitk.ImageSeriesReader()        
        dicom_names = reader.GetGDCMSeriesFileNames(indir)
        dicom_names = dicom_names[::-1]
        
        if not os.path.isfile(self.vol_file):
            reader.SetFileNames(dicom_names)
            itk_img = reader.Execute()
            vol = sitk.GetArrayFromImage(itk_img)
            print ("before",vol.shape)
            data=np.empty([vol.shape[0],1024,1024])
            for idx in range(vol.shape[0]):
                #print (idx,vol[idx].shape)
                array=np.require(vol[idx],np.int16,'c')
                data[idx]=cv2.resize(array,(1024,1024))
            vol=np.require(data,np.int32,'c')
            vol = vol.swapaxes(0,2)
            vol = vol.swapaxes(0,1)
            print ("after",vol.shape)
            
            # predict whether need to flip
        #    if flip_predict(vol):
        #        vol = vol[:,:,::-1]
        
        #    spacing = np.array(itk_img.GetSpacing())
            
            spacing = np.array(itk_img.GetSpacing())
            origin = np.array(itk_img.GetOrigin())
            
            vol_file = h5py.File(self.vol_file,'w-')
            vol_data = vol_file.create_dataset(self.id, data=vol, dtype=np.int16,
                                               chunks=True, compression='gzip')
            vol_data.attrs['spacing'] = spacing
            vol_data.attrs['origin'] = origin
            
            vol_file.close()
            
        else:
            vol_data = h5py.File(self.vol_file, 'r')
            vol = vol_data[self.id][...]
            spacing = vol_data[self.id].attrs['spacing']
        
        self.data = vol
        self.spacing = spacing
        self.filelists = dicom_names
#        return dict(data = vol, spacing = spacing)

    def seg_lung(self):
       
        origin = np.asarray(self.data.shape)
        target = np.asarray((256.,256.,self.data.shape[2]))
        data = zoom(self.data, target / origin, output=np.float, order=1, mode='nearest')
        spacing = self.spacing * origin/target
        
        lung = lung_segmentation_scan(data, spacing)
        lung = zoom(lung, origin/target, output=np.bool, order=1, mode='nearest')
            
        self.lungmask=lung
    
            
        lungseg_dir = os.path.join(self.cacheDir,'lungmask.pkl')
        with open(lungseg_dir, 'wb') as f:
                pickle.dump(self.lungmask, f, 2)
       

        #return lung
    
    def gen_list(self, candis):
        
        candidates_list = []
    
        for icandi in range(len(candis)):  # for every slice
            candi_slice = candis[icandi]
            if len(candi_slice) == 0:
                continue
    
            for bbox in candi_slice:  # for every candidate in one slice
                if bbox[4] < self.candis_thresh:
                    continue
                source_str = 'tmp'
                bb = [np.floor(bbox[1]), np.floor(bbox[0]), icandi, np.ceil(bbox[3])+1, np.ceil(bbox[2])+1, icandi+1]
                bb2 = np.array(bb, dtype=np.int)
                candidate_dict = dict(scanid=self.id, source=source_str, bbox=bb2)
                candidates_list.append(candidate_dict)
                
        self.candis = candidates_list
                
#        with open(os.path.join(self.cacheDir, 'candis.pkl'), 'wb') as f:
#            pickle.dump(candidates_list, f, -1)
    
    def cdd_filter_lungmask(self, thresh_r, candis):
    
        for icandi in range(len(candis)):  # for every slice
            candi_slice = candis[icandi]
            if len(candi_slice) == 0:
                continue
            _candi_slice = []
            for bbox in candi_slice: # for every candidate in one slice
                if np.max([(bbox[3]-bbox[1]) , (bbox[2]-bbox[0])] * self.spacing[0:2]) < 5:  # drop bbox with size less than 5mm
                    continue
                candi_r = max([(bbox[3]-bbox[1])/2, (bbox[2]-bbox[0])/2])
                candi_center = [int((bbox[1] + bbox[3]) / 2), int((bbox[0] + bbox[2]) / 2), icandi]
                if candi_r*self.spacing[0] < thresh_r:
                    if not self.lungmask[candi_center[0], candi_center[1], candi_center[2]]:
                        continue
                _candi_slice.append(bbox)
    
            candis[icandi] = _candi_slice
    
        return candis
        
    
    def final_predict(self):
        #self.lungmask = self.seg_lung()
        self.seg_lung()
        self.candis_thresh = 0.98
        
        all_bbx = test_net(net=self.net, vol=self.data)
        candis = self.cdd_filter_lungmask(thresh_r=10, candis=all_bbx)
        self.gen_list(candis)
        
        self.preprocess_data()
        config_dict = get_config(self.pkl_candis, self.vol_candis, batchsize=32, crop_shape=[64,64,64])
        valid_generator = config_dict['valid_generator']
        nb_valid_samples = config_dict['nb_valid_samples']
        model = fp_init.init(self.modelDir)
    
        probs = []
#        try:
        nb_seen_samples = 0
        while nb_seen_samples < nb_valid_samples:
            samples = next(valid_generator)#valid_generator.next()#
            probs.append(model.predict_on_batch(samples))
            nb_seen_samples += len(samples)
        assert nb_seen_samples == nb_valid_samples
        probs = np.concatenate(probs, axis=0)
    #        write_pkl_file(result_saveto, probs)
#        except KeyboardInterrupt:
#            pass

        self.probs = probs
#        return probs


    def preprocess_data(self):
        assert all(isinstance(c['bbox'], np.ndarray) and c['bbox'].shape==(6,) and c['bbox'].dtype==np.int for c in self.candis)
        assert all(np.all(c['bbox'] >= 0) & np.all(c['bbox'][-3:] > c['bbox'][:3]) for c in self.candis)
        
        
        self.pkl_candis = os.path.join(self.cacheDir, 'candis_final.pkl')
        self.vol_candis = os.path.join(self.cacheDir, 'vol_final.hdf5')
        
        self.candis = pipeline.main(self.candis, self.pkl_candis)
        
        if not os.path.isfile(self.vol_candis):       
            H5Writer(self.vol_candis, 'w',
                gen(
                    candidate_pkl_file=self.pkl_candis,
                    data_h5_files={
                        'tmp': self.vol_file,
                    },
                    min_edge_length=68.,
                )
            )
    
    def get_result(self):
        result_file = os.path.join(self.cacheDir, 'result.pkl')
        
        if not os.path.isfile(result_file):
            self.final_predict()
            self.postprocess_result()
            with open(result_file, 'wb') as f:
                pickle.dump(self.rlt, f, 2)
        else:
            with open(result_file, 'rb') as f:
                self.rlt = pickle.load(f)
    
    def postprocess_result(self):
        bbs = []
        for i in range(len(self.candis)):
            det =self.candis[i]
#            print self.probs.shape
            sc = self.probs[i][1]
            bbx = np.hstack((det["bbox"],sc))
#            cenr = np.hstack((det["center"],det["diameter"],sc))
            bbs.append(bbx)
        self.bbs=bbs
        self.candis_final=[]
        self.vol2slice(bbs)
            #print (candis)
        #print (len(self.candis_final))
        self.GetRlt(self.candis_final)
#        print (candis)
        
    def vol2slice(self, bbs):
        for i in range(len(bbs)):
            vol=bbs[i]
            #print (vol)
            z_st=np.int(vol[2])
            z_ed=np.int(vol[5])
            for z in range(z_st,z_ed):
                #self.candis.append([z,vol[0],vol[1],vol[3],vol[4],vol[6]])
                if vol[6]>=self.thr:
                    self.candis_final.append([z,vol[1],vol[0],vol[4],vol[3],vol[6]])
    
    def GetRlt(self,array):
        #print array
        #array=sio.loadmat("../data/02/candis2d.mat")["candis"]
        #print array
#        maxIdx=len(self.rlt)
        self.rlt=[np.array([0])]*len(self.filelists)
#        self.rlt=np.zeros(len(self.filelists))
#         print self.rlt
        for curRlt in array:
            print (curRlt)
            #idx=maxIdx-curRlt[0].astype(int)
            #idx=maxIdx-curRlt[0]
            idx=curRlt[0]
            #print (idx,curRlt[0].astype(int))
#            self.rlt[idx]=[self.rlt[idx],curRlt[1:6]]
            self.rlt[idx]=np.hstack((self.rlt[idx],curRlt[1:6]))
            print (self.rlt[idx])
            self.rlt[idx][0]+=1
            # mark the file name in the file list
#            self.textFileList.item(idx).setBackground(gui.QBrush(gui.QColor(0,0,255)))
            #print (self.rlt[curRlt[0].astype(int)],curRlt[0].astype(int))
    
    
    def loadFileInformation(self,indir):

        information = {}

        ds = pydicom.read_file(indir)    

        information['PatientID'] = ds.PatientID

        information['PatientName'] = ds.PatientName

        information['PatientBirthDate'] = ds.PatientBirthDate

        information['PatientSex'] = ds.PatientSex

        information['PatientAge'] = ds.PatientAge

        information['FileModDate'] = ds.FileModDate

        information['KVP'] = ds.KVP

        information['SliceThickness'] = ds.SliceThickness

        information['Width'] = ds.Width

        information['InstitutionName'] = ds.InstitutionName

        information['Manufacturer'] = ds.Manufacturer

        information['AccessionNumber'] = ds.AccessionNumber

        self.information = information

    def block_find(self,z):
        num = int(self.rlt[z][0])
        #self.data = self.data.astype(np.uint8)
        img =self.data[:,:,z].copy()
        try:
            lungseg_dir = os.path.join(self.cacheDir, 'lungmask.pkl')
            noduleseg_dir = os.path.join(self.cacheDir,'noduleseg.pkl')
            with open(lungseg_dir, 'rb') as f:
                self.lungmask = pickle.load(f)
            lungmask = self.lungmask
            # with open(noduleseg_dir,'rb') as g:
            #     mean = pickle.load(g)
            #     variance=pickle.load(g)
            #     contours_list=pickle.load(g)
            #lung = lung_segmentation_scan(data, spacing)
            #lungmask = zoom(lung, origin/target, output=np.bool, order=1, mode='nearest')
            
            #self.lungmask=lungmask
            lungmask = self.lungmask[:,:,z].copy()
        except:
            
                #
        # self.

            # lungmask = self.lungmask
            lungmask = self.lungmask[:,:,z].copy()
            # print lungmask
        tmp=np.zeros(lungmask.shape)
        tmp[lungmask[:, :] == False] = 0
        tmp[lungmask[:, :] == True] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tmp = cv2.erode(tmp, kernel)
        img[tmp[:,:] == 0] = -1200

        mask = []
        mean = []
        variance = []
        contours_list=[]
        for i in range (num):
            # img_ori = img[self.rlt[i][1]:self.rlt[i][3],self.rlt[i][2]:self.rlt[i][4]]
            img_ori = img[int(self.rlt[z][5*i+2]):int(self.rlt[z][5*i+4]),int(self.rlt[z][5*i+1]):int(self.rlt[z][5*i+3])]
            #img_ori = img[int(self.rlt[z][i][0]):int(self.rlt[z][i][2]),int(self.rlt[z][i][1]):int(self.rlt[z][i][3])]
            img_ori[img_ori < -1200] = -1200
            img_ori[img_ori > 300] = 300
            img_normalize = (img_ori+1200)*255./1500
            # tmpImg=np.require(img_ori,np.uint8,'c')
            #img_tmp = cv2.merge([img_ori,img_ori,img_ori])
            #img_ori = img_tmp.astype(np.uint8)
            msk,area = nodule_seg(img_normalize)
            # print msk.shape
            col = msk.shape[0]
            row = msk.shape[1]
            # print 'test', col, row
            valid_img = np.zeros([col, row])
            for m in range (col):
                for n in range (row):
                    if (msk[m,n] == 0):
                        valid_img[m,n]=0
                    else:
                        valid_img[m,n]=img_ori[m,n]
            # tmp = tmpImg[:,:]*1500/255.-1200
            valid_img = valid_img[valid_img.nonzero()]
            man = valid_img.mean()

            # print "mean: ", mean_of_CT
            # print "nodule's area: ",area

            #variance
            var = valid_img.var()
            
            # print msk.shape
            
            count = 0
            for i in range(col):
                for j in range(row):
                    if (msk[i,0] == 255):
                        count = count+1
                    if (msk[i,row-1] == 255):
                        count = count+1
                    if (msk[0,j] == 255):
                        count = count+1
                    if (msk[col-1,j] == 255):
                        count = count+1
            if (count >= (col+row)*3/2):
                msk[msk==255]=0
                man = 0
                var = 0
            contours,hierarchy = cv2.findContours(msk,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # print contours
            contours_list.append(contours)
            mask.append(msk)
            mean.append(man)
            variance.append(var)
    # edge.append(cv2.Canny(msk, 50, 150))
        
        contour_shape=[]
        for i in range (len(contours_list)):       
            contour_shape.append(np.array(contours_list[i]).shape)
        # print np.array(contours_list[i]).shape
    #print mean,variance    
    #print contour_shape
        #print contours_list
        # noduleseg_file={'mean':mean,'variance':variance,'contours_list':contours_list}
            # noduleseg_dir = os.path.join(self.cacheDir,'noduleseg.pkl')
            #with open(noduleseg_dir, 'wb') as f:
            #        pickle.dump(mean, f, 2)
            #        pickle.dump(variance, f, 2)
            #        pickle.dump(contours_list, f, 2)
         # self.build_data(self.modelDir)

        # print mean
        return mean,variance,contours_list

    
    
if __name__ == '__main__':
#    maindir = '/media/mjj/project/00.MyDSB/code/00.predict_end2end'
    indir = '/home/lxh/zrbTest/data/05'
    t1 = Predict(indir)

    t1.init_sys()
    t1.build_data(indir)
    t1.seg_lung()
    t1.final_predict()
    t1.get_result()
    # print t1.rlt
    for i in range (300):
            mean,variance,edge=t1.block_find(i)
            cv2.imshow(str(i),edge)
    # print mean
    # print variance

    print ("t1 done \n\n\n\n\n")

    #indir = '/home/ubuntu/zrbTest/data/05'
    #t2 = Predict(indir)
    #print t2.data.shape
    #t2.get_result()
    #print t2.rlt

    t1 = Predict(indir)

    t1.init_sys()
    t1.build_data(indir)
    t1.seg_lung()
    t1.final_predict()
    t1.get_result()
