
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import ResNet_quary
import ResNet_ref
from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        x = torch.tile(x, (1,3,1,1))
        y = torch.tile(y, (1,3,1,1))
        return 1. - super().forward(x, y)
    
def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


    
class Matching(nn.Module):
    def __init__(self, features1, features2):
        super(Matching, self).__init__()
        
    def forward(self, x):
        
        return 0
    
class CFOCNet(nn.Module):
    def __init__(self, args):
        super(CFOCNet, self).__init__()

        # self.resblock1_ref = ResidualBlock(3, 8)
        # self.resblock2_ref = ResidualBlock(8, 8)
        # self.resblock3_ref = ResidualBlock(8, 8)
        
        # self.resblock1_q = ResidualBlock(3, 8)
        # self.resblock2_q = ResidualBlock(8, 8)
        # self.resblock3_q = ResidualBlock(8, 8)
        
        
        
        self.self_attention1 = Self_Attn(256, nn.ReLU())
        self.self_attention2 = Self_Attn(256, nn.ReLU())
        self.self_attention3 = Self_Attn(256, nn.ReLU())
        
        self.conv_ref_1 = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        self.conv_ref_1_ = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        
        self.conv_ref_2 = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        self.conv_ref_2_ = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        
        self.conv_ref_3 = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        self.conv_ref_3_ = nn.Conv2d(64, 64, kernel_size=4, stride=2,padding=0)
        
        
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        
        self.bn4 = nn.BatchNorm2d(3)
        
        self.bn_S1 = nn.BatchNorm2d(1)
        self.bn_S2 = nn.BatchNorm2d(1)
        self.bn_S3 = nn.BatchNorm2d(1)
        
        self.bn_FS1 = nn.BatchNorm2d(1)
        self.bn_FS2 = nn.BatchNorm2d(1)
        self.bn_FS3 = nn.BatchNorm2d(1)
        
        self.insn_FS1 = nn.InstanceNorm2d(1)
        self.insn_FS2 = nn.InstanceNorm2d(1)
        self.insn_FS3 = nn.InstanceNorm2d(1)
        
        
        # self.max_pooling2d1 = nn.MaxPool2d(4, stride=4, padding=0)
        # self.max_pooling2d2 = nn.MaxPool2d(2, stride=2, padding=0)
        # self.max_pooling2d3 = nn.MaxPool2d(1, stride=1, padding=0)
        
        
        self.max_pooling2d1 = nn.AvgPool2d(4, stride=4, padding=0)
        self.max_pooling2d2 = nn.AvgPool2d(4, stride=4, padding=0)
        self.max_pooling2d3 = nn.AvgPool2d(4, stride=4, padding=0)
        
        
        self.GlobalMaxPooling = nn.AvgPool2d(64)
        
        # self.fix_conv2d1 = F.conv2d
        
        # self.conv1x1_1 = nn.Conv2d(1, 1, kernel_size=1, stride=1,padding=0)
        # self.conv1x1_2 = nn.Conv2d(1, 1, kernel_size=1, stride=1,padding=0)
        # self.conv1x1_3 = nn.Conv2d(1, 1, kernel_size=1, stride=1,padding=0)
        
        self.conv3x1 = nn.Conv2d(3, 1, kernel_size=1, stride=1,padding=0)
        # self.Softmax = nn.Softmax(dim=1)
        self.convt_1 = torch.nn.ConvTranspose2d(3, 1, kernel_size = 5, stride = 2,padding=0)
        self.convt_2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=5, stride = 2,padding=0)
        self.convt_3 = torch.nn.ConvTranspose2d(1, 1, kernel_size=5, stride = 2,padding=0)
        
        
        self.convt_s1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride = 1,padding=0)
        self.convt_s2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride = 1,padding=0)
        self.convt_s3 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride = 1,padding=0)
        
        self.upsampling1 = torch.nn.Upsample(size=(64,64), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling2 = torch.nn.Upsample(size=(64,64), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling3 = torch.nn.Upsample(size=(64,64), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        
        
        
        
        
        self.upsampling4 = torch.nn.Upsample(size=(128,128), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling5 = torch.nn.Upsample(size=(256,256), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        
        
        self.upsampling_rec_32 = torch.nn.Upsample(size=(32,32), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling_rec_64 = torch.nn.Upsample(size=(64,64), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling_rec_128 = torch.nn.Upsample(size=(128,128), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        self.upsampling_rec_256 = torch.nn.Upsample(size=(256,256), scale_factor=None, 
                                            mode='bilinear', align_corners=None)
        
        # fix_conv2d2 = F.conv2d(resblock2_q, weight)
        # fix_conv2d3 = F.conv2d(resblock3_q, weight)
        
        self.Sigmoid = nn.Sigmoid()
        self.Softmax2d = nn.Softmax2d()
        self.ReLU = nn.ReLU()
        self.tanh = nn.Tanh()
        self.Softmax = nn.Softmax()
        
        self.resnet_query = ResNet_quary.resnet50(pretrained=False, progress=True)
        
        self.resnet_ref = ResNet_ref.resnet50(pretrained=False, progress=True)
        
        self.classifier_conv1 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        self.classifier_linear = torch.nn.Linear(1*14*14, 80) 
        
        
        
        self.conv_redcce1_ref = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce2_ref = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce3_ref = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  
        
        self.conv_redcce1_ref2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce2_ref2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce3_ref2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  
        
        self.conv_redcce1_ref3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce2_ref3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce3_ref3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  
        
        self.bn_redcce1_ref = nn.BatchNorm2d(256)
        self.bn_redcce2_ref = nn.BatchNorm2d(256)
        self.bn_redcce3_ref = nn.BatchNorm2d(256)
        
        self.bn_redcce1_ref2 = nn.BatchNorm2d(256)
        self.bn_redcce2_ref2 = nn.BatchNorm2d(256)
        self.bn_redcce3_ref2 = nn.BatchNorm2d(256)
        
        self.bn_redcce1_ref3 = nn.BatchNorm2d(256)
        self.bn_redcce2_ref3 = nn.BatchNorm2d(256)
        self.bn_redcce3_ref3 = nn.BatchNorm2d(256)
        
        self.bn_redcce1_query = nn.BatchNorm2d(256)
        self.bn_redcce2_query = nn.BatchNorm2d(256)
        self.bn_redcce3_query = nn.BatchNorm2d(256)
        
        
        self.conv_redcce1_query = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce2_query = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv_redcce3_query = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        
        self.DropOut1 = nn.Dropout2d(p=0.2)
        self.DropOut2 = nn.Dropout2d(p=0.2)
        self.DropOut3 = nn.Dropout2d(p=0.2)
        
        self.conv1x1_1 = nn.Conv2d(3*1, 1, kernel_size=5, stride=1,padding=0)
        self.conv1x1_2 = nn.Conv2d(1, 1, kernel_size=5, stride=1,padding=0)
        self.conv1x1_3 = nn.Conv2d(1, 1, kernel_size=5, stride=1,padding=0)
        
        self.final_conv1 = nn.Conv2d(4, 1, kernel_size=3, stride=1,padding=1)
        
        
    
        
    def forward(self, query, ref, device):
        
        # print(query.shape)
        # print(ref.shape)
        
        out1_q, out2_q, out3_q = self.resnet_query(query)
        out1_ref, out2_ref, out3_ref = self.resnet_ref(ref)

        
        # print(out1_ref.shape)
        
        out1_q = self.conv_redcce1_query(out1_q)
        out1_q = self.bn_redcce1_query(out1_q)
        out1_q = self.ReLU(out1_q)
        
        out2_q = self.conv_redcce2_query(out2_q)
        out2_q = self.bn_redcce2_query(out2_q)
        out2_q = self.ReLU(out2_q)
        
        out3_q = self.conv_redcce3_query(out3_q)
        out3_q = self.bn_redcce3_query(out3_q)
        out3_q = self.ReLU(out3_q)


        # out1_q,_ = self.self_attention1(out1_q )
        # out2_q,_ = self.self_attention2(out2_q )
        # out3_q,_ = self.self_attention3(out3_q )

        ############################################
        out1_ref = self.conv_redcce1_ref(out1_ref)
        out1_ref = self.bn_redcce1_ref(out1_ref)
        out1_ref = self.tanh(out1_ref)
        
        out2_ref = self.conv_redcce2_ref(out2_ref)
        out2_ref = self.bn_redcce2_ref(out2_ref)
        out2_ref = self.tanh(out2_ref)
        
        out3_ref = self.conv_redcce3_ref(out3_ref)
        out3_ref = self.bn_redcce3_ref(out3_ref)
        out3_ref = self.tanh(out3_ref)
        ############################################
        
        # kernel = out1_ref1[0]
        # kernel = torch.transpose(kernel, 0,2).unsqueeze(3)
        # kernel = torch.reshape(kernel, (256, 16,16,-1))
        # print('111111', kernel.shape)
        # out1_ref = self.DropOut1(out1_ref)
        # out2_ref = self.DropOut2(out2_ref)
        # out3_ref = self.DropOut2(out3_ref)
        
        
        out1_ref = self.max_pooling2d1(out1_ref)
        out2_ref = self.max_pooling2d2(out2_ref)
        out3_ref = self.max_pooling2d3(out3_ref)
        
        
        # out1_ref1 = torch.unsqueeze(out1_ref1, dim=1)
        # out2_ref1 = torch.unsqueeze(out2_ref1, dim=1)
        # out3_ref1 = torch.unsqueeze(out3_ref1, dim=1)
        
        
        # out1_ref = torch.cat( (out1_ref1, out3_ref1, out3_ref1), dim=1 )
        # out2_ref = torch.cat( (out2_ref1, out2_ref2, out2_ref3), dim=1 )
        # out3_ref = torch.cat( (out3_ref1, out3_ref2, out3_ref3), dim=1 )
        
        

        # M1 = conv_query_ref(out1_q, out1_ref, device)
        # M2 = conv_query_ref(out2_q, out2_ref, device)
        # M3 = conv_query_ref(out3_q, out3_ref, device)
        

        q1 = out1_q.shape
        q2 = out2_q.shape
        q3 = out3_q.shape


        M1 = torch.zeros(q1[0], 1, q1[2]-16+1, q1[3]-16+1 , device=(torch.device(device)))
        M2 = torch.zeros(q2[0], 1, q2[2]-8+1, q2[3]-8+1 , device=(torch.device(device)))
        M3 = torch.zeros(q3[0], 1, q3[2]-4+1, q3[3]-4+1 , device=(torch.device(device)))

        



        for h in range(len(out1_q)):
            input_1 = out1_q[h,:,:,:].unsqueeze(0)
            kernel_1 = out1_ref[h,:,:,:].unsqueeze(0)
            # kernel_1 = torch.transpose(kernel_1, 3,0)
            
            
            input_2 = out2_q[h,:,:,:].unsqueeze(0)
            kernel_2 = out2_ref[h,:,:,:].unsqueeze(0)
            
            input_3 = out3_q[h,:,:,:].unsqueeze(0)
            kernel_3 = out3_ref[h,:,:,:].unsqueeze(0)
            
            
            # print('111111', kernel_1.shape)
            # print('111111', kernel_2.shape)
            # print('111111', kernel_3.shape)
            
            # print(input_1.shape)
            # print(kernel_1.shape)
            
            M = F.conv2d(input_1, kernel_1  ,stride= 1, padding=0)
            M = self.bn1(M)
            M1[h,:,:,:] = self.ReLU(M).squeeze(0)
            # M1[h,:,:,:] = M.squeeze(0)
            # print(M.shape)
            
            M = F.conv2d(input_2, kernel_2  ,stride=1, padding=0)
            M = self.bn2(M)
            M2[h,:,:,:] = self.ReLU(M).squeeze(0)
            # M2[h,:,:,:] = M.squeeze(0)
            
            M = F.conv2d(input_3, kernel_3 ,stride=1, padding=0)
            M = self.bn3(M)
            M3[h,:,:,:] = self.ReLU(M).squeeze(0)
            # M3[h,:,:,:] = M.squeeze(0)
            
        
        S1 = self.upsampling1(M1)
        S2 = self.upsampling2(M2)
        S3 = self.upsampling3(M3)
        
        # print(S1.shape)
        # print(S2.shape)
        # print(S3.shape)
        # S1 = F.interpolate(S1, ( 64, 64) )
        # S2 = F.interpolate(S2, ( 64, 64) )
        # S3 = F.interpolate(S3, ( 64, 64) )
        
        
        FS = torch.cat((S1,S2,S3), dim=1)
        
        # FS = self.bn4(FS)
        # FS = self.ReLU(FS)
        
        FS = self.conv1x1_1(FS)
        # FS = self.bn_FS1(FS)
        FS = self.ReLU(FS)
        FS = self.upsampling4(FS)
        FS = self.conv1x1_2(FS)
        FS = self.ReLU(FS)
        FS = self.upsampling5(FS)
        
        # FS = torch.cat( [FS, query], dim = 1 )
        # FS = self.final_conv1(FS)
        
        # FS1 = self.Sigmoid(FS)
        # FS = self.ReLU(FS)
        

        # FS = self.Sigmoid(FS)
        # Final_S = self.tanh(FS)
        # Final_S = self.Softmax2d(Final_S)
        
        
        return FS

class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive
        
def oneshot(model,img1,img2):
       # Gives you the feature vector of both inputs
       output1,output2 = model(img1.cuda(),img2.cuda())
       # Compute the distance 
       euclidean_distance = F.pairwise_distance(output1, output2)
       #with certain threshold of distance say its similar or not
       if euclidean_distance > 0.5:
               print("Orginal Signature")
       else:
               print("Forged Signature")
               
               


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
    

    

class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            
            nn.Conv2d(128, 256, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            
            nn.Conv2d(256, 256, 4),
            nn.ReLU(),   # 256@23*23
            nn.MaxPool2d(2), # 128@9*9
            
            nn.Conv2d(256, 256, 4),
            nn.ReLU(),   # 256@23*23
            nn.MaxPool2d(2), # 128@9*9
        )
        
        self.head1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4096, 256),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            )
            
        self.head2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            )
            
        self.head3 = nn.Sequential(
            nn.Linear(64, 2),
            nn.Tanh(),
            )

        # self.liner = nn.Sequential(nn.Dropout(p=0.5),
        #                            nn.Linear(4096, 1), 
        #                            nn.BatchNorm1d(1),
        #                            nn.Sigmoid())
        # self.out = nn.Linear(512, 1)

    def forward_one(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size()[0], -1)
        return feat

    def forward(self, x1):
        feat1 = self.forward_one(x1)
        # feat2 = self.forward_one(x2)
        # combined_features = feat1 * feat2
        
        
        combined_features = self.head1(feat1)
        combined_features = self.head2(combined_features)
        output =            self.head3(combined_features)
        
        # out = self.out(dis)
        #  return self.sigmoid(out)
        return output

    

# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss
#     Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
#     """

#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.eps = 1e-9

#     def forward(self, output1, output2, target, size_average=True):
#         distances = (output2 - output1).pow(2).sum(1)  # squared distances
#         losses = 0.5 * (target.float() * distances +
#                         (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
#         return losses.mean() if size_average else losses.sum()

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
    
    
    
    
def conv_query_ref(query, ref, device ):
    
    q = query.shape
    r = ref.shape
    w = r[2]
    h = r[3]
    # print(r)
    result = torch.zeros(q[0], 1, q[2], q[3] , device=(torch.device(device)))

    query = F.pad(input=query, pad=(int(w/2),int(w/2),int(h/2),int(h/2)), mode='constant', value=0)
    #(padding_left,padding_right, padding_top, padding_bottom)

    for i in range(q[2]):
        
        for j in range(q[3]):
            inpt1 = query[:,:,i:i+r[2],j:j+r[3]]
            out = torch.mul(inpt1,ref)
            result[:,0,i,j] = torch.sum(out,(1,2,3))
    
    return result
    
    
    
    
    
    
    

    
    
    
# Residual block
# query image: 256*256, with randomly flipped of probability 0.5
# reference images are resized to 64×64 with padding