import torch
import torch.nn as nn

from models.lora_llm import LORA
from models.fusion import Encoder,MultiHeadAttention,ScaledDotProductAttention


class TPCA(nn.Module):
    def __init__(self,dim1,dim2,d_k=64,d_v=64,n_head=8):
        super(TPCA,self).__init__()
        self.w1=nn.Linear(dim1,dim1)
        self.w2=nn.Linear(dim1,d_k*n_head)
        self.w3=nn.Linear(d_k*n_head,dim1)
        self.w4=nn.Linear(dim1,dim1)
        self.w5=nn.Linear(dim1,dim1)

        self.W_K = nn.Linear(dim2, d_k * n_head, bias=False)
        self.W_V = nn.Linear(dim2, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, dim2, bias=False)

        self.d_model = dim2
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_head)
        self.channel=dim1
    
    def forward(self,visual_features,text_features):
        batch_size,num,dim1=visual_features.size()
        residual=self.w5(visual_features)
        V_im=self.w1(visual_features)
        V_iq=self.w2(visual_features).reshape(batch_size,self.n_head,num,-1)
        L_ik=self.W_K(text_features).reshape(batch_size,self.n_head,-1,self.d_k)
        L_iv=self.W_V(text_features).reshape(batch_size,self.n_head,-1,self.d_k)
        context=self.ScaledDotProductAttention(V_iq,L_ik,L_iv)
        context=context.permute(0,2,1,3)
        context=context.reshape(batch_size,-1,self.n_head*self.d_v)
        context=context.permute(0,2,1)
        context=context.reshape(batch_size,num,self.n_head*self.d_v)
        context=self.w3(context)
        output=self.w4(V_im*context)
        return nn.LayerNorm(self.channel).cuda()(output + residual)
    
class GATE(nn.Module):
    def __init__(self,dim):
        super(GATE,self).__init__()
        self.lin1=nn.Linear(dim,64)
        self.lin2=nn.Linear(dim,64)
        self.attn1=MultiHeadAttention(64,d_k=64,d_v=64,n_heads=4,len_q=1,len_k=1)
        self.attn2=MultiHeadAttention(64,d_k=64,d_v=64,n_heads=4,len_q=1,len_k=1)
        self.lin_down=nn.Linear(50,25)

        self.lin3=nn.Linear(64,dim)
        self.lin4=nn.Linear(64,dim)
        self.dim=dim
    
    def forward(self,x,y):
        x=self.lin1(x)
        y=self.lin2(y)
        z=torch.cat([x,y],dim=-2)
        score1=self.attn1(x,z,z)
        score2=self.attn2(y,z,z)
        score1=self.lin3(score1)
        score2=self.lin4(score2)

        return nn.LayerNorm(self.dim).cuda()(score1+score2)
    
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
    
class Att3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax=nn.Softmax(dim=-2)
        self.relu=nn.ReLU()
        self.lin1=nn.Linear(dim,1)
        self.lin2=nn.Linear(dim,dim)

    def forward(self,input):
        #B,C,dim
        weights=self.softmax(self.lin1(input))
        out=weights*self.relu(self.lin2(input))
        return out

import torch.nn.functional as F
class CLIPLoss1(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cri1=nn.CrossEntropyLoss()
        self.cri2=nn.CrossEntropyLoss()
        

    def forward(self, image_features, text_features,labels):
        

        # L2归一化（余弦相似度等价于点积）
        image_features = F.normalize(image_features, dim=-1)
        image_features = image_features.squeeze(0)
        text_features = F.normalize(text_features, dim=-1)

        # 计算相似度矩阵
        logits_per_image = image_features @ text_features.T / self.temperature  # [batch_size, batch_size]
      
        # 创建标签（对角线为匹配对）
        # [0, 1, ..., batch_size-1]

        # 对称交叉熵损失
        loss_i2t = self.cri1(logits_per_image, labels)

       
        # 对称损失
        loss = loss_i2t
        return loss
    
class Prediction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1=nn.Linear(26,dim)
        self.lin2=nn.Linear(4,dim)
        self.lin3=nn.Linear(1024,dim)
        self.llm=LORA()
        self.enc=Encoder(dim)
        self.cri=CLIPLoss1()
        self.lin_cla1=nn.Linear(dim,30)
        self.lin_cla2=nn.Linear(30,21)
        enc_layers=nn.ModuleList([])
        self.vae=VAE(dim=dim)
        self.fu=GATE(dim=dim)
        for _ in range(12):
            enc_layers.append(TPCA(dim,dim))
        self.enc_layers=enc_layers

    
    def forward(self,answer,loc,visual):
        
        an=answer.unsqueeze(1)
        loc=loc.unsqueeze(1)
        B,_,_=an.size()
        an=self.lin1(an)
        loc=self.lin2(loc)
        input=self.fu(an,loc)
        visual=self.lin3(visual)
        visual=self.enc(visual)
        for layer in self.enc_layers:
            input=layer(input,visual)
        input=self.vae(input)
        out_final=self.lin_cla2(self.lin_cla1(input))
    
        return out_final