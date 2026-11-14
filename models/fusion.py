import torch
import torch.nn as nn
import numpy as np
from torch import einsum
import math
from einops import rearrange
import numpy as np
from transformers import VisualBertModel, VisualBertConfig
from models.TPCA import TPCA
from models.lora_llm import LORA
from timm import create_model

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
        z=torch.cat([x,y],dim=-2).permute(0,2,1)
        z=self.lin_down(z).permute(0,2,1)
        score1=self.attn1(x,z,z)
        score2=self.attn2(y,z,z)
        score1=self.lin3(score1)
        score2=self.lin4(score2)

        return nn.LayerNorm(self.dim).cuda()(score1+score2)

class Block(nn.Module):
    def __init__(self,dim):
        super(Block,self).__init__()
        self.att1=Encoder(dim)
        self.att2=Encoder(dim)
        self.tpca1=TPCA(dim,dim)
        self.tpca2=TPCA(dim,dim)
    def forward(self,x,y):
        x=self.att1(x)
        y=self.att2(y)
        x1=self.tpca1(x,y)
        y1=self.tpca2(y,x)

        return x1,y1
       
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=10000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        

        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[: ,:x.size(1), :].cuda() + x
        return pos
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads

    def forward(self, Q, K, V,mask=None):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) 
        if mask is not None:
            mask=mask.float()
            mask=mask.unsqueeze(1)
            attention_mask=torch.matmul(mask.permute(0,2,1),mask)
            attention_mask = attention_mask.cuda()
            attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-1e10')) # scores : [batch_size, n_heads, len_q, len_k]
    
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
         # [batch_size, n_heads, len_q, d_v]
        return context
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads)
        self.len_q = len_q
        self.len_k = len_k
       

    def forward(self, input_Q, input_K, input_V,mask=None):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        #print("Q.SIZE",Q.size())
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context= self.ScaledDotProductAttention(Q, K, V,mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


"""
FeedForward
"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)

class Encoder(nn.Module):
    def __init__(self,d_model=768,d_ff=2048, d_k=64, d_v=64, n_heads=8, len_q=1):
        super(Encoder,self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self,enc_input,mask=None):
        enc_outputs= self.enc_self_attn(enc_input, enc_input, enc_input,mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Fusion(nn.Module):
    def __init__(self,dim,n_heads,num_TPCA):
        super(Fusion,self).__init__()
        self.attn1=Encoder(n_heads=n_heads)
        self.attn2=Encoder(n_heads=n_heads)
        
        self.lin_projection2=nn.Linear(1024,dim)
        self.TPCA=nn.ModuleList()
        for _ in range(num_TPCA):
            self.TPCA.append(TPCA(dim1=dim,dim2=dim))
        self.LLM=LORA()
        self.num=num_TPCA
        self.pe=FixedPositionalEncoding(embedding_dim=768)
        self.deit = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin')) 

        config=VisualBertConfig.from_pretrained("/media/cygzz/data/guanjl1/surgical_VQLA/Surgical-VQLA/models/uclanlp/visualbert-vqa-coco-pre")
        config.hidden_size = dim
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

    def forward(self,visual,inputs):
        B,S,d=visual.size()
        input_ids=inputs['input_ids'].cuda()
        token_type_ids=inputs['token_type_ids'].cuda()
        visual_token_type_ids = torch.ones(visual.shape[:-1], dtype=torch.long).cuda()
        visual_embeds=visual
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        visual_embeds=self.lin_projection2(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        text_out=self.attn1(embeddings)
        visual_out=self.attn2(visual_embeddings)
        visual_in=visual_out
        for layer in self.TPCA:
            fusion_out=layer(visual_in,text_out)
            visual_in=fusion_out

        #fusion_embedding = self.deit.blocks(fusion_out)
        

        return fusion_out


class Fusion1(nn.Module):
    def __init__(self,dim,n_heads,num_TPCA):
        super(Fusion1,self).__init__()
        self.attn1=Encoder(n_heads=n_heads)
        self.attn2=Encoder(n_heads=n_heads)
        
        self.lin_projection2=nn.Linear(512,dim)
        self.TPCA=nn.ModuleList()
        for _ in range(num_TPCA):
            self.TPCA.append(Block(dim))
        self.fusion=GATE(dim)
        self.LLM=LORA()
        self.num=num_TPCA
        self.pe=FixedPositionalEncoding(embedding_dim=768)
        self.deit = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin')) 

        config=VisualBertConfig.from_pretrained("/media/cygzz/data/guanjl1/surgical_VQLA/Surgical-VQLA/models/uclanlp/visualbert-vqa-coco-pre")
        config.hidden_size = dim
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

    def forward(self,visual,inputs):
        B,S,d=visual.size()
        input_ids=inputs['input_ids'].cuda()
        token_type_ids=inputs['token_type_ids'].cuda()
        visual_token_type_ids = torch.ones(visual.shape[:-1], dtype=torch.long).cuda()
        visual_embeds=visual
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        visual_embeds=self.lin_projection2(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        
        for layer in self.TPCA:
            embeddings,visual_embeddings=layer(embeddings,visual_embeddings)
            
        fusion_embedding = self.deit.blocks(self.fusion(embeddings,visual_embeddings))
        

        return fusion_embedding


class Fusion2(nn.Module):
    def __init__(self,dim,n_heads,num_TPCA):
        super(Fusion2,self).__init__()
        self.attn1=Encoder(n_heads=n_heads)
        self.attn2=Encoder(n_heads=n_heads)
        
        self.lin_projection2=nn.Linear(512,dim)
        self.TPCA=nn.ModuleList()
        for _ in range(num_TPCA):
            self.TPCA.append(Block(dim))
        self.fusion=GATE(dim)
        self.LLM=LORA()
        self.num=num_TPCA
        self.pe=FixedPositionalEncoding(embedding_dim=768)
        self.deit = create_model("deit_base_patch16_224", pretrained=True,pretrained_cfg_overlay=dict(file='/media/cygzz/data/guanjl1/surgical_VQLA/EndoVisLoc/models/pytorch_model.bin')) 

        config=VisualBertConfig.from_pretrained("/media/cygzz/data/guanjl1/surgical_VQLA/Surgical-VQLA/models/uclanlp/visualbert-vqa-coco-pre")
        config.hidden_size = dim
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.visual_token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.visual_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        if config.special_visual_initialize:
            self.visual_token_type_embeddings.weight.data = nn.Parameter(
                self.token_type_embeddings.weight.data.clone(), requires_grad=True
            )
            self.visual_position_embeddings.weight.data = nn.Parameter(
                self.position_embeddings.weight.data.clone(), requires_grad=True
            )

    def forward(self,visual,inputs):
        B,S,d=visual.size()
        input_ids=inputs['input_ids'].cuda()
        token_type_ids=inputs['token_type_ids'].cuda()
        visual_token_type_ids = torch.ones(visual.shape[:-1], dtype=torch.long).cuda()
        visual_embeds=visual
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings

        position_embeddings = self.position_embeddings(position_ids)
        embeddings += position_embeddings

        visual_embeds=self.lin_projection2(visual_embeds)
        visual_token_type_embeddings = self.visual_token_type_embeddings(visual_token_type_ids)
        visual_position_ids = torch.zeros(
            *visual_embeds.size()[:-1], dtype=torch.long, device=visual_embeds.device
        )
        visual_position_embeddings = self.visual_position_embeddings(visual_position_ids)
        visual_embeddings = visual_embeds + visual_position_embeddings + visual_token_type_embeddings
        
        for layer in self.TPCA:
            embeddings,visual_embeddings=layer(embeddings,visual_embeddings)
            
        fusion_embedding = self.fusion(embeddings,visual_embeddings)
        

        return fusion_embedding



        
        

   

if __name__ == "__main__":
    from transformers import BertTokenizer
    
    questions=['What is the state of prograsp_forceps?']
    visual_features=torch.ones((1,25,512))
    visual_features = visual_features.cuda()
    tokenizer = BertTokenizer.from_pretrained('/media/cygzz/data/guanjl1/surgical_VQLA/Surgical-VQLAPlus/dataset/bertvocab_extended_infu/v2/')
    inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=25)
    model=Fusion(dim=768,n_heads=8,num_TPCA=4)
    model=model.cuda()
    out=model(visual_features,inputs)
    print("finish")