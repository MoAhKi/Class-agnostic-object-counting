# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 13:42:05 2021

@author: Mohammad
"""

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tifffile import imsave


def Myloss( output, target):
    loss = torch.pow((target - output), 2)
    loss = torch.mul(loss, target)
    return loss.mean()

# def se_block(in_block, ch, ratio=16):
#     x = GlobalAveragePooling2D()(in_block)
#     x = Dense(ch//ratio, activation='relu')(x)
#     x = Dense(ch, activation='sigmoid')(x)
#     return multiply()([in_block, x])


def CountingLoss(a,b):
    
    return np.average(abs(a-b))

def Evaluation(args, dataloader_, device, net,criterion_SSIM,criterion_E):
    TestCounter = 0
    print("Evaluation...")
    # net.eval()
    
    countig_loss = []
    
    for i, (query, ref, Target) in enumerate(dataloader_,0):
            
            query, ref, Target = map(lambda x: x.to(device), [query, ref,Target])
            pred_target = net(query.float(), ref.float(), device)
            
            # loss_SSIM = criterion_SSIM(pred_target, Target)
            # print(loss2.shape)
            loss_E = criterion_E(pred_target, Target) 
            
            loss = loss_E#+ 1e-3 *loss_SSIM
            
            query = query.detach().to('cpu').numpy()
            ref = ref.detach().to('cpu').numpy()
            Target = Target.detach().to('cpu').numpy()
            pred_target = pred_target.detach().to('cpu').numpy()
            
            
            real_num_obj = np.sum(Target, (1,2,3))
            pred_num_obj = np.sum(pred_target, (1,2,3))
            
            countig_loss.append( CountingLoss(pred_num_obj, real_num_obj ) )

            
            for s in range(len(query)):

                sample1 = query[s,:,:,:]
                sample1 = np.transpose(sample1, (1, 2, 0))
                
                sample2 = ref[s,0:3,:,:]
                sample2 = np.transpose(sample2, (1, 2, 0))
                
                sample3 = Target[s,:,:,:]
                sample3 = np.transpose(sample3, (1, 2, 0))
                
                sample4 = pred_target[s,:,:,:]
                sample4 = np.transpose(sample4, (1, 2, 0))
    
                
                img1 = query[s,:,:,:]
                img2 = ref[s,0:3,:,:]
                img3 = Target[s]
                img4 = pred_target[s]
                
                input_img = np.concatenate( (img1,img2) , 2 )
                input_img = np.transpose(input_img, (1, 2, 0))
                filename = 'figs/figs_test/input_img_%d.png'%(TestCounter)
                imsave(filename, input_img)
                
                # Big_img = np.concatenate( (img3,img4) , 2 ).squeeze(0)
                
                plt.clf()
                plt.subplot( 1,2,1 )
                plt.imshow(img3.squeeze(0))
                plt.subplot( 1,2,2 )
                plt.imshow(img4.squeeze(0))
                filename = 'figs/figs_test/Big_img_%d.png'%(TestCounter)
                plt.show()
                plt.savefig(filename)
                plt.pause(0.01)
    
                TestCounter +=1
            
            
            if TestCounter >= 64:
                break
    print("Done")
    return loss.detach().to('cpu').numpy(), np.average(countig_loss)
    
    