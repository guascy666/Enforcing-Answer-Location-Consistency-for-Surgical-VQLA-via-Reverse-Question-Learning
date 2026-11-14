import torch
import torch.nn.functional as F

def DualFocalLoss(input, target):
    gamma = 1.0
    if input.dim()>2:
        input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
    target = target.view(-1,1)

    logp_k = F.log_softmax(input, dim=1)
    softmax_logits = logp_k.exp()
    logp_k = logp_k.gather(1, target)
    logp_k = logp_k.view(-1)
    p_k = logp_k.exp()  # p_k: probility at target label
    p_j_mask = torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1  # mask all logit larger and equal than p_k
    p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

    loss = -1 * (1 - p_k + p_j) ** gamma * logp_k

    return loss.mean()

def clip_contrastive_loss(global_features,local_features, temperature=0.07):
    """
    CLIP的图文对比损失函数（仅对角线为正样本）
    Args:
        image_features: 图像特征矩阵，shape [batch_size, feature_dim]（已归一化）
        text_features: 文本特征矩阵，shape [batch_size, feature_dim]（已归一化）
        temperature: 温度系数，CLIP默认0.07
    Returns:
        total_loss: 图像视角损失 + 文本视角损失的平均值
        img_loss: 图像视角的对比损失
        text_loss: 文本视角的对比损失
    """
    # 确保特征归一化（CLIP的图像/文本编码器输出会先归一化）
    global_features=global_features.squeeze(1)
    local_features=local_features.squeeze(1)
    global_features = F.normalize(global_features, dim=-1)
    local_features = F.normalize(local_features, dim=-1)
    
    # 1. 计算图文跨模态相似度矩阵 [batch_size, batch_size]
    # sim_i2t[i,j] = 第i张图与第j段文本的余弦相似度
    sim_i2t = torch.matmul(global_features, local_features.t()) / temperature  # [N, N]
    # sim_t2i[j,i] = 第j段文本与第i张图的余弦相似度（与sim_i2t对称）
    sim_t2i = sim_i2t.t()  # [N, N]
    
    # 2. 构建标签：每个样本的正样本是对角线位置（自身配对的图文）
    batch_size = global_features.shape[0]
    labels = torch.arange(batch_size).to(global_features.device)  # [0,1,...,N-1]
    
    # 3. 计算图像视角损失：每个图像找对应文本（正样本）
    img_loss = F.cross_entropy(sim_i2t, labels)
    # 4. 计算文本视角损失：每个文本找对应图像（正样本）
    text_loss = F.cross_entropy(sim_t2i, labels)
    
    # 5. 总损失为两个视角损失的平均
    total_loss = (img_loss + text_loss) / 2
    
    return total_loss


def cla_loss(t1,s1,label,temporature=10,alpha=0.7):
    t_soft=F.softmax(t1/temporature,dim=-1)
    s_soft1=F.softmax(s1/temporature,dim=-1)
    
    distill_loss=F.kl_div(s_soft1.log(),t_soft,reduction='batchmean')*(temporature**2)
    cls_loss=DualFocalLoss(s1,label)
    return alpha*distill_loss+(1-alpha)*cls_loss
 
