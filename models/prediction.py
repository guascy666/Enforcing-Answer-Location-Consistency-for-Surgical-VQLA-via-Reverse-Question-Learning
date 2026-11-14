import torch
import torch.nn as nn
from models.fusion import *
from timm import create_model
from models.classifier import Prediction as classifier
class Answer(nn.Module):
    def __init__(self,dim,num_classes):
        super(Answer,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, int(dim//2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(int(dim//2),num_classes )
        )

    def forward(self,input):
        return self.classifier(input)

class Location(nn.Module):
    def __init__(self,dim):
        super(Location,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, int(dim//2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(int(dim//2),4 )
        )

    def forward(self,input):
        return self.classifier(input)
    
class VAE(nn.Module):
    def __init__(self,dim=512):
        super(VAE,self).__init__()
        self.lin=nn.Linear(dim,1)
        self.lin2=nn.Linear(dim,dim)
        self.relu=nn.ReLU()
        self.act=nn.Softmax(dim=-1)

    
    def forward(self,features):
        #(B.N.C)
        weight=self.relu(self.lin(features))#(B,N,1)
        weight=self.act(weight.permute(0,2,1))
        features=(self.lin2(features))
        output=torch.matmul(weight,features)

        return output
    
class Prediction(nn.Module):
    def __init__(self, dim,n_heads,num_TPCA,num_reg,num_class):
        super(Prediction,self).__init__()
        self.backbone=Fusion(dim,n_heads,num_TPCA)
        self.answer=nn.ModuleList([])
        self.location=nn.ModuleList([])

        for i in range(num_reg):
            self.answer.append(Encoder())
            self.location.append(Encoder())
        self.vae_an=VAE(dim)
        self.vae_loc=VAE(dim)
        self.vae_adv=VAE(dim)
        self.lin_an=Answer(dim,num_class)
        self.lin_loc=Location(dim)
        self.name='Prediction'
        self.deit1 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))
        self.deit2 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))  
        self.classifier=classifier(dim=64)
        self.classifier.load_state_dict(torch.load('/media/cygzz/data/guanjl1/surgical_VQLA/EXP4_A2Q/results/head_8_TPCA_12_lr_0.0001/best.pth'))
        
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def A2Q(self,an,bbox,gt_an,gt_bbox,visual):
        self.classifier.eval()
        out1=self.classifier(an,bbox,visual)
        out2=self.classifier(gt_an,gt_bbox,visual)
    
        
        return out1,out2


        
        
    
    def forward(self,inputs,visual):
        fusion_embedding=self.backbone(visual,inputs)
        avd_output=self.vae_adv(fusion_embedding)
        answer_embedding=self.deit1.blocks(fusion_embedding)
        location_embedding=self.deit2.blocks(fusion_embedding)
        answer_embedding=self.vae_an(answer_embedding)
        location_embedding=self.vae_loc(location_embedding)
        answer=self.lin_an(answer_embedding)
        bbox=self.lin_loc(location_embedding).sigmoid()

        return answer.squeeze(1),bbox.squeeze(1),avd_output
    

class Prediction1(nn.Module):
    def __init__(self, dim,n_heads,num_TPCA,num_reg,num_class):
        super(Prediction1,self).__init__()
        self.backbone=Fusion1(dim,n_heads,num_TPCA)
        self.answer=nn.ModuleList([])
        self.location=nn.ModuleList([])
        self.name='Prediction1'

        self.vae_an=VAE(dim)
        self.vae_loc=VAE(dim)
        # self.lin_an=nn.Sequential(nn.Linear(dim,int(dim//2)),nn.Linear(int(dim//2),num_class))
        # self.lin_loc=nn.Sequential(nn.Linear(dim,int(dim//2)),nn.Linear(int(dim//2),4))

        self.lin_an=Answer(dim,num_class)
        self.lin_loc=Location(dim)
        
    
    def forward(self,inputs,visual):
        fusion_embedding=self.backbone(visual,inputs)
        answer_embedding=fusion_embedding
        location_embedding=fusion_embedding

        
        
        answer_embedding=self.vae_an(answer_embedding)
        location_embedding=self.vae_loc(location_embedding)
        answer=self.lin_an(answer_embedding)
        bbox=self.lin_loc(location_embedding).sigmoid()
     
        
        # answer_embedding=self.vae_an(answer_embedding)
        # location_embedding=self.vae_loc(location_embedding)
        # answer=self.lin_an(answer_embedding)
        # bbox=self.lin_loc(location_embedding).sigmoid()

        # answer_embedding=fusion_embedding.mean(dim=-2)
        # location_embedding=fusion_embedding.mean(dim=-2)
        # answer=self.lin_an(answer_embedding)
        # bbox=self.lin_loc(location_embedding).sigmoid()

        return answer.squeeze(1),bbox.squeeze(1)

    

class Prediction2(nn.Module):
    def __init__(self, dim,n_heads,num_TPCA,num_reg,num_class):
        super(Prediction2,self).__init__()
        self.backbone=Fusion2(dim,n_heads,num_TPCA)
        self.answer=nn.ModuleList([])
        self.location=nn.ModuleList([])
        self.name='Prediction2'

        self.vae_an=VAE(dim)
        self.vae_loc=VAE(dim)
        self.lin_an=nn.Sequential(nn.Linear(dim,int(dim//2)),nn.Linear(int(dim//2),num_class))
        self.lin_loc=nn.Sequential(nn.Linear(dim,int(dim//2)),nn.Linear(int(dim//2),4))

        # self.lin_an=Answer(dim,num_class)
        # self.lin_loc=Location(dim)

        self.deit1 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))
        self.deit2 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))  
        
    
    def forward(self,inputs,visual):
        fusion_embedding=self.backbone(visual,inputs)
        answer_embedding=self.deit1.blocks(fusion_embedding)
        location_embedding=self.deit2.blocks(fusion_embedding)
     
        
        answer_embedding=self.vae_an(answer_embedding)
        location_embedding=self.vae_loc(location_embedding)
        answer=self.lin_an(answer_embedding)
        bbox=self.lin_loc(location_embedding).sigmoid()

        # answer_embedding=answer_embedding.mean(dim=-2)
        # location_embedding=location_embedding.mean(dim=-2)

        # answer=self.lin_an(answer_embedding)
        # bbox=self.lin_loc(location_embedding).sigmoid()

        return answer.squeeze(1),bbox.squeeze(1)

from models.swin_transformer_v2 import SwinTransformerV2

    
class Prediction_eval(nn.Module):
    def __init__(self, dim=768,n_heads=8,num_TPCA=10,num_reg=2,num_class=26):
        super(Prediction_eval,self).__init__()
        self.backbone=Fusion(dim,n_heads,num_TPCA)
        self.answer=nn.ModuleList([])
        self.location=nn.ModuleList([])

        for i in range(num_reg):
            self.answer.append(Encoder())
            self.location.append(Encoder())
        self.vae_an=VAE(dim)
        self.vae_loc=VAE(dim)
        self.lin_an=Answer(dim,num_class)
        self.lin_loc=Location(dim)
        self.name='Prediction'
        self.deit1 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))
        self.deit2 = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin'))  
        self.swin=SwinTransformerV2()
        
    
 
    def forward(self,inputs,img):
        visual=self.swin.forward(img)
        fusion_embedding=self.backbone(visual,inputs)
    
        answer_embedding=self.deit1.blocks(fusion_embedding)
        location_embedding=self.deit2.blocks(fusion_embedding)
        answer_embedding=self.vae_an(answer_embedding)
        location_embedding=self.vae_loc(location_embedding)
        answer=self.lin_an(answer_embedding)
        bbox=self.lin_loc(location_embedding).sigmoid()

        return answer.squeeze(1),bbox.squeeze(1)
