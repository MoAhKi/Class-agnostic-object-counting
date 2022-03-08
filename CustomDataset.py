# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 12:38:06 2021
@author: Mohammad
"""
import os
import numpy as np
import torch
import glob as glob
import time
from PIL import Image
from torchvision import transforms
import torchvision.datasets as datasets
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader,Dataset
from random import randrange
import torch.nn.functional as F


def makeGaussian2(x_center=0, y_center=0, theta=0, sigma_x = 10, sigma_y=10, x_size=640, y_size=480):
    # x_center and y_center will be the center of the gaussian, theta will be the rotation angle
    # sigma_x and sigma_y will be the stdevs in the x and y axis before rotation
    # x_size and y_size give the size of the frame 

    # theta = 2*np.pi*theta/360
    x = np.arange(0,x_size, 1, float)
    y = np.arange(0,y_size, 1, float)
    y = y[:,np.newaxis]
    sx = sigma_x
    sy = sigma_y
    x0 = x_center
    y0 = y_center

    # rotation
    # a=np.cos(theta)*x -np.sin(theta)*y
    # b=np.sin(theta)*x +np.cos(theta)*y
    # a0=np.cos(theta)*x0 -np.sin(theta)*y0
    # b0=np.sin(theta)*x0 +np.cos(theta)*y0

    return np.exp(-(((x-x0)**2)/(2*(sx**2)) + ((y-y0)**2) /(2*(sy**2))))

CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
           'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
           'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
           'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
           'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class CountingkDataset(Dataset):
    
    def __init__(self,path ,NumberOfObjects):
        super(CountingkDataset, self).__init__()
        self.path = path
        self.feedshape_ref = [3, 256, 256]
        # self.feedshape_query = [3, 473, 473]
        self.feedshape_query = [3, 256, 256]
        self.feedshape_target = [1, 256, 256]
        
        self.image_paths = glob.glob(os.path.join(self.path+"imgs_bbox/", "*.png"))
        # self.image_paths = self.image_paths [:100]
        
        self.image_labels = glob.glob(os.path.join(self.path+"imgs_bbox/"+"*.txt"))
        print('Number of imgs: %d'%(len(self.image_paths)))
        
        self.indices = np.arange(len(self.image_paths))
        self.NumberOfObjects = NumberOfObjects

        #,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                   # transforms.Resize(self.feedshape_query[1:]),
                                               
        self.transform_query = transforms.Compose([transforms.ToTensor() ])
        
        self.transform_ref = transforms.Compose([transforms.ToTensor() ])
        
        self.transform_target = transforms.Compose([transforms.ToTensor() ])
        
        self.number_RandomKernel = 10
    
    def crop_pad_to_size(self, image, Target , desire_h, desire_w):
        """ zero-padding a four dim feature matrix: N*C*H*W so that the 
            new Height and Width are the desired ones desire_h and desire_w 
            should be largers than the current height and weight
        """
        
        cur_h = image.shape[-2]
        cur_w = image.shape[-1]
        ##  Croping
        right = 0
        left = 0
        top = 0
        bottom = 0
        if cur_h > 256:
            x1 = randrange(0, cur_h - 256)
            image = image[:,x1: x1 + 256, :]
            Target = Target[:,x1: x1 + 256,:]
        
        
        if cur_w > 256:
            y1 = randrange(0, cur_w - 256)
            image = image[:,:, y1: y1 + 256]
            Target = Target[:,:, y1: y1 + 256]
            
        ## Padding
        cur_h = image.shape[-2]
        cur_w = image.shape[-1]
        
        if cur_h < 256:
            top_pad = (desire_h - cur_h + 1) // 2
            bottom_pad =(desire_h - cur_h) - top_pad
            image = F.pad(image, (0, 0, top_pad, bottom_pad))
            Target = F.pad(Target, (0, 0, top_pad, bottom_pad))
            
        if cur_w < 256:
            left_pad = (desire_w - cur_w + 1) // 2
            right_pad = (desire_w - cur_w) - left_pad
            image = F.pad(image, (left_pad, right_pad, 0, 0))
            Target = F.pad(Target, (left_pad, right_pad, 0, 0))
        
        # print("$$$$$$$$$$")
        # print(image.shape)
        # print("$$$$$$$$$$")

        return image, Target 

    def __getitem__(self,index):
        # address = self.path + "imgs_bbox/" +"image_"+ str(int(index)).zfill(10)+".png"
        # print(index)
        # img = Image.open(address)
        img = Image.open(self.image_paths[index])
        img_shape = np.shape(img)
        # img = self.transform(img).float()
        
        
        
        label = np.loadtxt(self.image_labels[index])
        shape= np.shape(label)
        indice = np.arange(int(shape[0]))
        idx = random.choice(indice)  ## to Select object
        
        query_img_idx = label[idx,4]
        
        # print(CLASSES[int(query_img_idx)])
        
        path = self.path+"objects/"+str(int(query_img_idx)).zfill(2)+"/"+ "*.png"
        ListOfCondidate = glob.glob(os.path.join(path))
        numberOfRef = 5
        numberOfRefCounter = 0
        
        for numberOfRefCounter in range(numberOfRef):
            if numberOfRefCounter == 0:
                
                ref = random.choice(ListOfCondidate) 
            
                Ref = Image.open(ref)
                Ref = Ref.resize(( 64 , 
                                  64), 
                                  Image.ANTIALIAS)
            else:
                ref = random.choice(ListOfCondidate)
                ref = Image.open(ref)
                ref = ref.resize(( 64,
                                  64),
                                  Image.ANTIALIAS)
                
                Ref = np.concatenate((Ref, ref), axis=2 )
                
        # ref = ref.resize(( self.feedshape_ref[1] , 
        #                   self.feedshape_ref[2]), 
        #                   Image.ANTIALIAS)
        
        
        Ref = np.tile(Ref,(4,4,1))
        
        targets_in_img =  label[label[:, 4]== query_img_idx, :]
        
        
        Target = []
        for count, bbox in enumerate(targets_in_img):
            x1 = int(round(bbox[1]))
            x2 = int(round(bbox[3]))
            y1 = int(round(bbox[0]))
            y2 = int(round(bbox[2]))
            # print(x1, x2, y1, y2)
            target = makeGaussian2(x_center= (y2+y1)/2, 
                          y_center=  (x2+x1)/2, 
                          sigma_x = 10, 
                          sigma_y= 10, 
                          x_size=img_shape[1], 
                          y_size=img_shape[0])
            
            target = target/target.sum()
            target = np.expand_dims(target, 0)
            Target.append(target)
            
        Target = np.asarray(Target)
        Target = np.sum(Target, 0)
        Target = np.transpose(Target, (1,2,0))
        
        
        query = self.transform_query(img)
        Target = self.transform_target(Target)
        Ref = self.transform_ref(Ref)
        
        query , Target = self.crop_pad_to_size(query , Target,  256, 256)
        
        return query, Ref, Target
    
    def __len__(self):
        return len(self.image_paths)
    
    
class SiameseNetworkDataset(Dataset):
    
    def __init__(self,path ,transform=True):
        self.path = path    
        self.feedshape = [3, 100, 100]
        self.image_paths = glob.glob(os.path.join(self.path, "*.png"))
        print('Number of imgs: %d'%(len(self.image_paths)))
        self.labels = np.load(self.path+"/bbox.npy")
        self.indice = np.arange(len(self.image_paths))
        print(self.labels.shape)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize(self.feedshape[1:])])
    def __getitem__(self,index):
        
        idx0 = random.choice(self.indice)
        label0 = self.labels[idx0,4]
        
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                idx1 = random.choice(self.indice)
                if label0 == self.labels[idx1,4]:
                    break
        else:
            idx1 = random.choice(self.indice)
        
        label1 = self.labels[idx1,4]
        # img0 = Image.open(self.image_paths[idx0]).convert("RGB")
        # img1 = Image.open(self.image_paths[idx1]).convert("RGB")
        
        img0 = Image.open(self.image_paths[idx0])
        img1 = Image.open(self.image_paths[idx1])

        # img0 = img0.convert("L")
        # img1 = img1.convert("L")        

        img0 = self.transform(img0).float()
        img1 = self.transform(img1).float()

        # return img0, img1, torch.FloatTensor([label0 == label1 ])
        return img0, img1, torch.FloatTensor([int(label0 == label1 )])
    
    def __len__(self):
        return len(self.image_paths)
    