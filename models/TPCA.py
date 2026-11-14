import torch.nn as nn
import torch
from torch import einsum
import math
from einops import rearrange
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
    
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
         # [batch_size, n_heads, len_q, d_v]
        return context

class Att3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax=nn.Softmax(dim=-2)
        self.relu=nn.ReLU()
        self.lin1=nn.Linear(dim,1)
        self.lin2=nn.Linear(dim,dim)
        self.lin_down=nn.Linear(50,25)

    def forward(self,input1,input2):
        input=torch.cat([input1,input2],dim=-2)

        weights=self.softmax(self.lin1(input))
        out=weights*self.relu(self.lin2(input))

        out=out.permute(0,2,1)
        out=self.lin_down(out)
        out=out.permute(0,2,1)
        return out






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
        self.att3=Att3(dim=dim1)
    
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
        out_final=nn.LayerNorm(self.channel).cuda()(output + residual)
   
        
        return out_final


    
if __name__ == "__main__":
    import argparse
    from transformers import BertTokenizer
    questions=['What is the state of prograsp_forceps?']
    visual_features=torch.ones((2,25,512)).cuda()
    visual_features = visual_features.cuda()
    tokenizer = BertTokenizer.from_pretrained('/media/cygzz/data/guanjl1/surgical_VQLA/Surgical-VQLAPlus/dataset/bertvocab_extended_infu/v2/')
    inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)
    model=TPCA(512,768,64,64,8)
    mode=model.cuda()
    text_features=torch.ones((2,25,768)).cuda()
    out=model(visual_features,text_features)
    print("finish")