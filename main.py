


"""
    Experiment: Multi kernel in reference network
    


"""


import os
import argparse
import numpy as np
import torch
import random 
import cv2
from model import *
from CustomDataset import *

import json
import torchvision.datasets as dset
# from gluoncv import data, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.datasets as datasets
import pytorch_ssim
import glob as glob
from datetime import datetime

from utils import *
from tifffile import imsave


IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    parser = argparse.ArgumentParser(description='CFOCNet')
   
    parser.add_argument("--epochs",dest= 'epochs', default= 200) 
   
    parser.add_argument("--ngpu",dest= 'ngpu', default= 1)
   
    parser.add_argument("--batch_size", dest = "batch_size", help = "Batch size", 
                        default = 4)
    
    parser.add_argument("--learning_rate", dest = "learning_rate", 
                        help = "learning_rate", default = 0.000001)
    
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    
    parser.add_argument("--train_path", dest = 'train_path', help = 
                        "train_path",
                        default = "E:/datasets/Train_Images2017/", 
                        type = str)
    
    parser.add_argument("--test_path", dest = 'test_path', help = 
                        "test_path",
                        default = "E:/datasets/Test_Images2017/", 
                        type = str)
    return parser.parse_args()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.002)
        torch.nn.init.xavier_uniform(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant


if __name__ == '__main__':
    print('Main')
    args = arg_parse()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # BatchLoader_ = BatchLoader(args.train_path , 80)
    
    train_dataset   = CountingkDataset(args.train_path, 80)
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size= args.batch_size, 
                                  pin_memory= False,
                                  shuffle = True ,
                                  num_workers= 7)
    
    test_dataset   = CountingkDataset(args.test_path, 80)
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size= args.batch_size, 
                                  pin_memory= True,
                                  shuffle = False)

    
    # train_dataset   = CountingkDataset(args.train_path, 80)
    # train_dataloader = DataLoader(train_dataset, 
    #                               batch_size= args.batch_size, 
    #                               pin_memory= False,
    #                               shuffle = True)
    
    # test_dataset   = CountingkDataset(args.test_path, 80)
    # test_dataloader = DataLoader(test_dataset, 
    #                               batch_size= args.batch_size, 
    #                               pin_memory= True,
    #                               shuffle = False)
    
    net = CFOCNet(args).to(device)
    
    # net.apply(weights_init)
    weights_normal_init(net, dev=0.0001)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    
    # criterion = ContrastiveLoss(margin = 0.1)
    # criterion_E = torch.nn.L1Loss().cuda(device)
    # criterion_E = torch.nn.L2Loss().cuda(device)
    criterion_E = nn.MSELoss().cuda(device)
    # criterion_E2 = nn.L1Loss().cuda(device) 
    # criterion = nn.CrossEntropyLoss()
    # criterion = ContrastiveLoss(margin = 2.0)
    criterion_SSIM = SSIMLoss().cuda(device) # .cuda() if you need GPU support
    
    criterion_cls = nn.CrossEntropyLoss().cuda(device)
    # criterion_cls = nn.MSELoss().cuda(device)
    
    # myloss = Myloss()
    
    if  os.path.isfile('SavedModel/net_checkpoint.pth'):
        print('Loading checkpoint...')
        FILE_NET = 'SavedModel/net_checkpoint.pth'
        net.load_state_dict(torch.load(FILE_NET))
        net.eval()
        print('Done')

        
    for name, param in net.resnet_query.named_parameters():
        param.requires_grad = True
        # print(param.requires_grad)
    
    for name, param in net.resnet_ref.named_parameters():
        param.requires_grad = True
        # print(param.requires_grad)  

    loss_history = [] 
    loss_history_test = []
    loss_counting_test = []
    counter= []
    iteration_number = 0
    
    path_imgs = glob.glob(os.path.join(args.train_path+"imgs_bbox/", "*.png"))
    path_labels = glob.glob(os.path.join(args.train_path+"imgs_bbox/"+"*.txt"))
    
    data_length = len( path_imgs )
    

    for epoch in range(args.epochs):
        for i, (query, ref, Target) in enumerate(train_dataloader,0):
            # start=datetime.now()
            optimizer.zero_grad()
            # print(iteration_number)
            query, ref, Target= map(lambda x: x.to(device), 
                                                    [query, ref, Target])
            
            # print(np.shape(ref))
            # print(np.shape(query))
            # print(np.shape(query))
            
            pred_target = net(query.float(), ref.float(), device)

            # loss_SSIM = criterion_SSIM(pred_target, Target)
            # print(loss2.shape)
            # loss_E = torch.sum(torch.mul(Target, criterion_E(pred_target, Target)))
            loss_E = criterion_E(pred_target.float(), Target.float())
            # loss_E2 = criterion_E2(pred_target, Target)
                        
            # loss = loss_E + 1e-3 *loss_SSIM
            # myloss =  Myloss(pred_target, Target)
            # loss = loss_SSIM + 1e-2*loss_E
            
            # loss_cls = criterion_cls( query_img_idx_pred,  query_img_idx)
            
            # loss_cls = criterion_cls( query_img_idx_pred,  query_img_idx)
            
            # TotalLoss = loss + loss_cls
            TotalLoss = loss_E#+ 1e-3 *loss_SSIM
            # TotalLoss = loss_alltarget+ loss_E
            
            TotalLoss.backward()
            optimizer.step()
            
            iteration_number +=1
            
            # print(datetime.now()-start)
            
            if iteration_number % 50 == 0 :
                real_num_obj = np.sum(Target.detach().to('cpu').numpy(), (1,2,3))
                pred_num_obj = np.sum(pred_target.detach().to('cpu').numpy(), (1,2,3))
                
                counting_loss = CountingLoss(pred_num_obj, real_num_obj )
                
                print("Epoch: {}, {}/{}, counting_loss: {:.3f}".format(epoch, 
                                                     i, 
                                                     len(train_dataloader),
                                                     counting_loss))
                counter.append(iteration_number)
                loss_history.append(TotalLoss.detach().to('cpu').numpy())
                np.savetxt('results/loss_history.csv', loss_history, delimiter=',')
            
            if iteration_number % 1000 == 0 :
                
                query = query.detach().to('cpu').numpy()
                sample1 = query[0,:,:,:]
                sample1 = np.transpose(sample1, (1, 2, 0))
                
                
                ref = ref.detach().to('cpu').numpy()
                sample2 = ref[0,0:3,:,:]
                # print(np.shape(sample2))
                sample2 = np.transpose(sample2, (1, 2, 0))
                
                Target = Target.detach().to('cpu').numpy()
                sample3 = Target[0,:,:,:]
                sample3 = np.transpose(sample3, (1, 2, 0))
                
                
                pred_target = pred_target.detach().to('cpu').numpy()
                sample4 = pred_target[0,:,:,:]
                sample4 = np.transpose(sample4, (1, 2, 0))
                
                
                img1 = query[0,:,:,:]
                img2 = ref[0,0:3,:,:]
                img3 = Target[0]
                img4 = pred_target[0]
                
                
                
                input_img = np.concatenate( (img1,img2) , 2 )
                input_img = np.transpose(input_img, (1, 2, 0))
                
                filename = 'figs/figs_train/input_img_%d.png'%(iteration_number)
                imsave(filename, input_img)
                
                # Big_img = np.concatenate( (img3,img4) , 2 ).squeeze(0)
                
                
                
                plt.clf()
                plt.subplot( 1,2,1 )
                plt.imshow(img3.squeeze(0))
                plt.subplot( 1,2,2 )
                plt.imshow(img4.squeeze(0))
                filename = 'figs/figs_train/Big_img_%d.png'%(iteration_number)
                plt.show()
                plt.savefig(filename)
                plt.pause(0.01)
                
                """ Evaluation on test dataset
                """
            
            if iteration_number % 5000 == 0 :
                
                optimizer.zero_grad()
                loss_test, countig_loss = Evaluation(args, 
                                                     test_dataloader,
                                                     device,
                                                     net,
                                                     criterion_SSIM,
                                                     criterion_E )
                
                loss_history_test.append(loss_test)
                loss_counting_test.append(countig_loss)
                np.savetxt('results/loss_history_test.csv', loss_history_test, 
                            delimiter=',')
                np.savetxt('results/loss_counting_test.csv', loss_counting_test, 
                            delimiter=',')
                net.train()
                
            if iteration_number % 5000 == 0 :
                PATH_NET = 'SavedModel/net_checkpoint.pth'
                torch.save(net.state_dict(), PATH_NET)
                
                