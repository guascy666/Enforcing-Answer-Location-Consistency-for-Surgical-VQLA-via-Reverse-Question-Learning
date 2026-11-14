'''
Description     : Train model.
Paper           : CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
'''

import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data  import DataLoader
import logging
from utils import *
from dataloader import *
from models.prediction import Prediction,Prediction1,Prediction2
from models.con_loss import clip_contrastive_loss
from tqdm import tqdm
import warnings
from models.con_loss import cla_loss
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
Seed randoms
'''

question_all=['What is the state of bipolar_forceps?',
 'What is the state of clip_applier?',
 'What is the state of large_needle_driver?',
 'What is the state of monopolar_curved_scissors?',
 'What is the state of prograsp_forceps?', 'What is the state of stapler?',
 'What is the state of suction?', 'What is the state of ultrasound_probe?',
 'What is the tool in the left_bottom?',
 'What is the tool in the left_top?',
 'What is the tool in the right_bottom?',
 'What is the tool in the right_top?', 'What organ is being operated?',
 'Where is bipolar_forceps located?','Where is clip_applier located?',
 'Where is large_needle_driver located?',
 'Where is monopolar_curved_scissors located?',
 'Where is prograsp_forceps located?', 'Where is stapler located?',
 'Where is suction located?', 'Where is ultrasound_probe located?']


# 分类结果
question_organ = [
    'What organ is being operated?'
]

question_location = [
    'Where is bipolar_forceps located?',
    'Where is clip_applier located?',
    'Where is large_needle_driver located?',
    'Where is monopolar_curved_scissors located?',
    'Where is prograsp_forceps located?',
    'Where is stapler located?',
    'Where is suction located?',
    'Where is ultrasound_probe located?'
]


question_organ = [
    'What organ is being operated?'
]

question_location = [
    'Where is clip_applier located?',
    'Where is large_needle_driver located?',
    'Where is monopolar_curved_scissors located?',
    'Where is prograsp_forceps located?',
    'Where is stapler located?',
    'Where is suction located?',
    'Where is ultrasound_probe located?'
]

question_instrument = [
    'What is the tool in the left_bottom?',
    'What is the tool in the left_top?',
    'What is the tool in the right_bottom?',
    'What is the tool in the right_top?',
    'Where is bipolar_forceps located?',
]

question_operation = [
    'What is the state of bipolar_forceps?',
    'What is the state of clip_applier?',
    'What is the state of large_needle_driver?',
    'What is the state of monopolar_curved_scissors?',
    'What is the state of prograsp_forceps?',
    'What is the state of stapler?',
    'What is the state of suction?',
    'What is the state of ultrasound_probe?'
]



def seed_everything(seed=0):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    
    model.train()
    awl = AutomaticWeightedLoss(4)  
    loss_func = ContrastiveLoss(batch_size=args.batch_size, device=device, temperature=0.05)
    total_loss = 0.0    
    total_loss_class = 0.0
    total_loss_bbox = 0.0
    epsilon = 0.04
    label_true = None
    label_pred = None
    label_score = None
    bbox_outputs_pred = None
    bbox_label_true = None
    


    for i, (_, visual_features, q, labels, bbox_label,local_features) in enumerate(tqdm(train_dataloader),0):

        # prepare questions
        questions = []
        # for i in range(len(q[0])): 
        #     for j in range(len(q)):
        #         questions.append(q[j][i])
        
        for question in q:
            questions.append(question)
        visual_features = visual_features.to(device)
        visual_features = visual_features.view(-1,36,1024)
        local_features = local_features.to(device)
        local_features = local_features.view(-1,36,1024)
        local_features.requires_grad=False
        labels = labels.to(device)
        labels = labels.view(-1)
        bbox_label = bbox_label.to(device)
        bbox_label = bbox_label.view(-1,4)
        label_embedding=F.one_hot(labels,num_classes=args.num_class)
        label_embedding=label_embedding.float()
        label_embedding=label_embedding.detach()
        bbox_embedding=bbox_label.detach()

        inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)
        

        
        
        (classification_outputs, bbox_outputs,adv_output_global) = model(inputs, visual_features)  
        (_,_,adv_output_local) = model(inputs, local_features)  

        loss_con=clip_contrastive_loss(adv_output_global,adv_output_local)
        
        classification_1=F.softmax(classification_outputs,dim=-1)
        s,t,=model.A2Q(classification_1,bbox_outputs,label_embedding,bbox_embedding,visual_features) 
        s=s.squeeze()
        t=t.squeeze()
    
        

        label_new=[]

        for question in q:
                label_new.append(question_all.index(question))
                
        label_new=torch.tensor(label_new).cuda()

        loss_a2q=cla_loss(t,s,label_new)


        

        if args.claloss == 'ce':
            loss_class = criterion(classification_outputs, labels)
        else:
            loss_class = DualFocalLoss(classification_outputs, labels)

        if args.detloss == 'giou':
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True, GIoU=True).sum()
        elif args.detloss == 'l1':   
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True).sum()
        elif args.detloss == 'l1g':
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True, l1g=True).sum()
        elif args.detloss == 'ciou':
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True, CIoU=True).sum() 
        elif args.detloss == 'diou':
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True, DIoU=True).sum()


        if args.uncer:
            loss = awl(loss_class, loss_bbox)   ## uncertainty loss     
        else:    
            loss = loss_class + loss_bbox
        
        loss=loss+float(args.w_cla)*loss_a2q+float(args.w_con)*loss_con

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
     

        # print statistics
        total_loss += loss.item()
        # total_loss += loss.item()
        total_loss_class += loss_class.item()
        total_loss_bbox += loss_bbox.item()

        scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
        label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
        label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
        label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
        bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
        bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
        
    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    bbox_miou, std_iou = mIoU_xyxy(box_cxcywh_to_xyxy(bbox_label_true), box_cxcywh_to_xyxy(bbox_outputs_pred))
    
    print('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f| mIoU: %.6f| Std_IoU: %.6f' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou, std_iou))
    os.makedirs(result_dir,exist_ok=True)
    base_dir='head_'+str(args.n_heads)+'_TPCA_'+str(args.num_TPCA)+'_w_cla_'+str(args.w_cla)+'_w_con_'+str(args.w_con)+'_lr_'+str(args.lr)
    os.makedirs(os.path.join(result_dir,base_dir),exist_ok=True)
    with open(os.path.join(result_dir,base_dir) + '.txt','a') as f:    
        f.write('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f| mIoU: %.6f| Std_IoU: %.6f \n' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou, std_iou))  
    return acc

labels_all = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False,name='18'):

    
    counts_pre = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }

    counts_gt = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }

    pro_pre = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }

    pro_gt = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }


    bbox_pre = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }

    bbox_gt = {
        '组织类': None,
        '操作类': None,
        '方位类': None,
        '器械类': None,
        '未分类': None  # 兜底：统计不在四大类的标签
    }
    
    model.eval()
    awl = AutomaticWeightedLoss(4)  
    total_loss = 0.0    
    total_loss_class = 0.0
    total_loss_bbox = 0.0
    label_true = None
    label_pred = None
    label_score = None
    bbox_outputs_pred = None
    bbox_label_true = None
    file_names = list()
    pro_gt_all=None
    pro_pre_all=None

    
    
    criterion = nn.CrossEntropyLoss()
    
    
    with torch.no_grad():
        for i, (file_name, visual_features, q, labels, bbox_label) in enumerate(tqdm(val_loader),0):
            # prepare questions
            questions = []
            for question in q: questions.append(question)
            inputs = tokenizer(questions, return_tensors="pt", padding="max_length", max_length=args.question_len)

            # GPU / CPU
            visual_features = visual_features.to(device)
            labels = labels.to(device)
            bbox_label = bbox_label.to(device)
            label_embedding=F.one_hot(labels,num_classes=args.num_class)

            
                                

            ( classification_outputs, bbox_outputs,_) = model(inputs, visual_features)  
            loss_class = DualFocalLoss(classification_outputs, labels)
            loss_bbox = bbox_iou(bbox_outputs, bbox_label, xywh=True, GIoU=True).sum()
            loss = awl(loss_class, loss_bbox) 
            
            # print statistics
            total_loss += loss.item()
            total_loss_class += loss_class.item()
            total_loss_bbox += loss_bbox.item()
            
            scores, predicted = torch.max(F.softmax(classification_outputs, dim=1).data, 1)    
            label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
            bbox_outputs_pred = bbox_outputs.data.cpu() if bbox_outputs_pred == None else torch.cat((bbox_outputs_pred, bbox_outputs.data.cpu()), 0)
            bbox_label_true = bbox_label.data.cpu() if bbox_label_true == None else torch.cat((bbox_label_true, bbox_label.data.cpu()), 0)
            current_bbox_pred = bbox_outputs.data.cpu()
            current_bbox_gt = bbox_label.data.cpu()

            pro_pre_all = F.softmax(classification_outputs, dim=1).data.cpu() if pro_pre_all  == None else torch.cat((pro_pre_all, F.softmax(classification_outputs, dim=1).data.cpu()), 0)
            pro_gt_all = label_embedding.data.cpu() if  pro_gt_all == None else torch.cat((pro_gt_all, label_embedding.data.cpu()), 0)

            for kk in range(len(questions)):
                tag=questions[kk]

                curr_label = labels[kk].data.cpu().unsqueeze(0)
                curr_pred = predicted[kk].data.cpu().unsqueeze(0)
                curr_bbox_gt = current_bbox_gt[kk].unsqueeze(0)  # 取当前batch的第kk个bbox gt
                curr_bbox_pred = current_bbox_pred[kk].unsqueeze(0)
                curr_pro_gt = label_embedding[kk].data.cpu().unsqueeze(0)  # 取当前batch的第kk个bbox gt
                curr_pro_pred = F.softmax(classification_outputs, dim=1)[kk].data.cpu().unsqueeze(0)  # 取当前batch的第kk个bbox pred

                # 按类别统计
                if tag in question_organ:
                    category = '组织类'
                elif tag in question_instrument:
                    category = '器械类'
                elif tag in question_location:
                    category = '方位类'
                elif tag in question_operation:
                    category = '操作类'
                else:
                    category = '未分类'
                
                # 拼接当前样本到对应类别（统一逻辑，避免重复代码）
                counts_gt[category] = curr_label if counts_gt[category] is None else torch.cat((counts_gt[category], curr_label), 0)
                counts_pre[category] = curr_pred if counts_pre[category] is None else torch.cat((counts_pre[category], curr_pred), 0)
                bbox_gt[category] = curr_bbox_gt if bbox_gt[category] is None else torch.cat((bbox_gt[category], curr_bbox_gt), 0)
                bbox_pre[category] = curr_bbox_pred if bbox_pre[category] is None else torch.cat((bbox_pre[category], curr_bbox_pred), 0)

                pro_gt[category] = curr_pro_gt if pro_gt[category] is None else torch.cat((pro_gt[category], curr_pro_gt), 0)
                pro_pre[category] = curr_pro_pred if pro_pre[category] is None else torch.cat((pro_pre[category], curr_pro_pred), 0)

            for n in range(len(questions)):
                item_pred=predicted[n].data.cpu()
                item_bbox=bbox_outputs[n].data.cpu()
                if item_pred==labels[n].data.cpu():
                    flag='YES'
                else:
                    flag='NO'
                iou_item,_=mIoU_xyxy(box_cxcywh_to_xyxy(torch.cat((bbox_label[n].unsqueeze(dim=0).data.cpu(),bbox_label[n].unsqueeze(dim=0).data.cpu()),0)), box_cxcywh_to_xyxy(torch.cat((item_bbox.unsqueeze(dim=0),item_bbox.unsqueeze(dim=0)),0)))
                line=file_name[n]+'|'+questions[n]+'|'+labels_all[item_pred]+'|'+flag+'|'+str(box_cxcywh_to_xyxy(item_bbox).tolist())+'|'+str(iou_item)+'\n'
                with open(name+ '_all_case.txt','a') as f:    
                    f.write(line)


                        
            
            
            #for f in file_name: file_names.append(f)
    
    print('dataset:',name)
    print("="*80)
    print(f"{'类别':<8} {'准确率(Acc)':<12} {'F1分数(F-Score)':<15} {'mIoU':<12} {'IoU标准差':<12} {'consistency':<12}")
    print("-"*80)
    for category, num in counts_gt.items():
        if counts_gt[category]==None:
            continue
        acc_item=calc_acc(counts_gt[category],counts_pre[category])
        _,_, fscore_item = calc_precision_recall_fscore(counts_gt[category],counts_pre[category])
        bbox_miou_item, std_iou_item= mIoU_xyxy(box_cxcywh_to_xyxy(bbox_gt[category]), box_cxcywh_to_xyxy(bbox_pre[category]))
        consistency_item=calculate_mixed_score(pro_pre[category],pro_gt[category],box_cxcywh_to_xyxy(bbox_gt[category]), box_cxcywh_to_xyxy(bbox_pre[category]))
        print(f"{category:<8} {acc_item:.4f} {fscore_item:.4f} {bbox_miou_item:.4f} {std_iou_item:.4f} {consistency_item:4f}")
            
    acc = calc_acc(label_true, label_pred) 
    c_acc = 0.0
    # c_acc = calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    bbox_miou, std_iou= mIoU_xyxy(box_cxcywh_to_xyxy(bbox_label_true), box_cxcywh_to_xyxy(bbox_outputs_pred))    
    consistency=calculate_mixed_score(pro_pre_all,pro_gt_all,box_cxcywh_to_xyxy(bbox_label_true), box_cxcywh_to_xyxy(bbox_outputs_pred))
    # with open(args.robust_output, 'a') as file:
    #     file.write('Test: Acc: %.6f | mIoU: %.6f \n' %(acc, bbox_miou))
    print('dataset:',name)
    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f| mIoU: %.6f| Std_IoU: %.6f' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou, std_iou))
    print('consistency:',consistency)
    result_dir='results'
    os.makedirs(result_dir,exist_ok=True)
    base_dir='head_'+str(args.n_heads)+'_TPCA_'+str(args.num_TPCA)+'_w_cla_'+str(args.w_cla)+'_w_con_'+str(args.w_con)+'_lr_'+str(args.lr)
    os.makedirs(os.path.join(result_dir,base_dir),exist_ok=True)
    with open(os.path.join(result_dir,base_dir) + '.txt','a') as f:    
        f.write('dataset: '+name)
        f.write('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f |C-Score: %.6f\n' %(epoch, total_loss, acc, precision, recall, fscore, bbox_miou,consistency))                 

    if save_output:
        '''
            Saving predictions
        '''
        bbox_outputs_pred_org = box_cxcywh_to_xyxy(bbox_outputs_pred * torch.tensor([1280, 1024, 1280, 1024])) + torch.tensor([1, 1, 1, 1])
        bbox_label_true_org = box_cxcywh_to_xyxy(bbox_label_true * torch.tensor([1280, 1024, 1280, 1024])) + torch.tensor([1, 1, 1, 1])
        if os.path.exists(args.checkpoint_dir + 'text_files') == False:
            os.mkdir(args.checkpoint_dir + 'text_files' ) 
        file1 = open(args.checkpoint_dir + 'text_files/labels.txt', 'w')
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + 'text_files/predictions.txt', 'w')
        file1.write(str(label_pred))
        file1.close()

        if args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction", "Bbox Ground Truth", "Bbox Prediction", "mIoU"])
        single_bbox_miou=[None] * len(label_true)
        for i in range(len(label_true)):
            single_bbox_miou[i] = mIoU_single(box_cxcywh_to_xyxy(bbox_label_true[i]), box_cxcywh_to_xyxy(bbox_outputs_pred[i]))
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]],
                           'Bbox Ground Truth': bbox_label_true_org[i], 'Bbox Prediction': bbox_outputs_pred_org[i],
                           'mIoU': single_bbox_miou[i]}, ignore_index=True)
        
        df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore, bbox_miou,consistency)

def get_log(file_name):
    logger = logging.getLogger('*')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionLocalizedAnswering')
    
    # Model parameters
    parser.add_argument('--n_heads',        type=int,   default=8,                                  help='Multi-head attention.')
    parser.add_argument('--w_cla',        type=float,   default=0.1,                                  help='Multi-head attention.')
    parser.add_argument('--w_con',        type=float,   default=0.1,                                  help='Multi-head attention.')
    parser.add_argument('--reg',        type=int,   default=2,                                  help='Multi-head attention.')
    parser.add_argument('--num_TPCA',        type=int,   default=12,                                  help='Multi-head attention.')
    parser.add_argument('--epochs',         type=int,   default=80,                                 help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--lr',             type=float, default=0.00005,                            help='0.000005, 0.00001, 0.000005')
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/cat/',                            help='/vb/vbrm/cat/') 
    parser.add_argument('--dataset_type',   default= 'endovis',                                     help='endovis')
    parser.add_argument('--patch_size',     type=int, default= 5,                                   help='1/2/3/4/5')
    parser.add_argument('--question_len',   type=int, default= 36,                                  help='25')
    parser.add_argument('--num_class',      type=int, default= 2,                                   help='26')
    parser.add_argument('--validate',       default=False,                                          help='When only validation required False/True')
    parser.add_argument('--detloss',        default=None,                                           help='detloss')
    parser.add_argument('--claloss',        default=None,                                           help='claloss')
    parser.add_argument('--uncer',          default=None,                                           help='uncer')
    parser.add_argument('--robustpath',     default= '',                                            help='')
    parser.add_argument('--robust_output',     default= '',                                            help='')
    parser.add_argument('--CAL',            default=None,                                           help='CAL')
    
    args = parser.parse_args()

    seed_everything()
    print(args.validate)
    
    # GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # best model initialize
    start_epoch = 1
    best_epoch = [0,0,0]
    best_results = [0.0,0.0,0.0]
    epochs_since_improvement = 0
    
    best_accuracy = 0
    best_accuracy_epoch = 0
    best_miou = 0
    best_miou_epoch = 0
    best_avg_acc_miou = 0
    best_avg_acc_miou_epoch = 0
    
    # dataset
    if args.dataset_type == 'endovis':
        '''
        Train and test dataloader for EndoVis18 & EndoVis17
        '''
        # tokenizer
        tokenizer = BertTokenizer.from_pretrained('dataset/bertvocab_extended_infu/v2/')
        # tokenizer = BertTokenizer.from_pretrained('dataset/bertvocab/v2/bert-EndoVis-18-VQA/')


        # data location
        
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        
        folder_head = '/media/cygzz/data/guanjl1/surgical_VQLA/datasets/EndoVis-18-VQLA_new/VQLA/seq_'
        folder_tail = '/vqla/*.txt'
          
        # dataloader
        train_dataset = EndoVis18VQAClassification(train_seq, folder_head, folder_tail, patch_size = args.patch_size,mode='train')
        train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True)
        val_dataset = EndoVis18VQAClassification(val_seq, folder_head, folder_tail, patch_size = args.patch_size, robustpath=args.robustpath,mode='valid')
        val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False)
        
        # external dataloader
        external_folder_head = '/media/cygzz/data/guanjl1/surgical_VQLA/datasets/EndoVis-17-VQLA_new/EndoVis-17-VQLA'
        external_folder_tail = '/vqla/*.txt'        
        external_val_dataset = EndoVis17VQAClassificationValidation(external_folder_head, external_folder_tail, patch_size = args.patch_size, robustpath=args.robustpath)
        external_val_dataloader = DataLoader(dataset=external_val_dataset, batch_size= args.batch_size, shuffle=False)

        SGH_val_dataset=SGHVQAClassificationValidation('/media/cygzz/data/guanjl1/surgical_VQLA/datasets/SGH_VQLA/VQLA','/*.txt',patch_size=5)
        SGH_val_dataloader=DataLoader(dataset=SGH_val_dataset,batch_size= args.batch_size)

        # num_classes
        args.num_class = 26

    # Initialize / load checkpoint
    if args.checkpoint is None:
        # model
        
        model = Prediction(dim=768,n_heads=args.n_heads,num_TPCA=args.num_TPCA,num_reg=args.reg,num_class=args.num_class)
        print(model.name)
        
        
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_Acc = checkpoint['Acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        final_args = checkpoint['final_args']
        for key in final_args.keys(): args.__setattr__(key, final_args[key])

    # Move to GPU, if available
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)

    # Loss function for classification head
    criterion = nn.CrossEntropyLoss().to(device)


    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir,exist_ok=True)


    # validation
    if args.validate:
        # with open(args.robust_output, 'a') as file:
        #     file.write(args.robustpath)
        result_dir='results'
        base_dir='head_'+str(args.n_heads)+'_TPCA_'+str(args.num_TPCA)+'_w_cla_'+str(args.w_cla)+'_w_con_'+str(args.w_con)+'_lr_'+str(args.lr)
        

        result_dir1='results_eval'
        os.makedirs(result_dir1,exist_ok=True)
        base_dir='head_'+str(args.n_heads)+'_TPCA_'+str(args.num_TPCA)+'_w_cla_'+str(args.w_cla)+'_w_con_'+str(args.w_con)+'_lr_'+str(args.lr)

        checkpoint_path =  checkpoint_path = os.path.join(result_dir,base_dir,'best_18.pth')
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        test_acc, test_c_acc, test_precision, test_recall, test_fscore, test_bbox_miou,consistency = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device,name='endovis18')
        with open(result_dir1 + '.txt','a') as f:  
            f.write('Setting:'+base_dir+'\n')
            f.write('Dataset:'+'EndoVis-18'+'\n')  
            f.write(' Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f|C-Score: %.6f \n' %(test_acc,  test_precision, test_recall, test_fscore, test_bbox_miou,consistency)) 
        
        checkpoint_path =  checkpoint_path = os.path.join(result_dir,base_dir,'best_17.pth')
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        ext_test_acc, ext_test_c_acc, ext_test_precision, ext_test_recall, ext_test_fscore, ext_test_bbox_miou,consistency = validate(args, val_loader=external_val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device,name='endovis17')
        with open(result_dir1 + '.txt','a') as f:  
            f.write('Dataset:'+'EndoVis-17'+ '\n')  
            f.write(' Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f|C-Score: %.6f\n' %(ext_test_acc, ext_test_precision, ext_test_recall, ext_test_fscore, ext_test_bbox_miou,consistency))
        
        checkpoint_path =  checkpoint_path = os.path.join(result_dir,base_dir,'best_SGH.pth')
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        ext_test_acc1, ext_test_c_acc1, ext_test_precision1, ext_test_recall1, ext_test_fscore1, ext_test_bbox_miou1,consistency = validate(args, val_loader=SGH_val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device,name='SGH')
        with open(result_dir1 + '.txt','a') as f:  
            f.write('Dataset:'+'SGH'+ '\n')  
            f.write(' Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f | mIoU: %.6f |C-Score: %.6f\n' %(ext_test_acc1, ext_test_precision1, ext_test_recall1, ext_test_fscore1, ext_test_bbox_miou1,consistency))
            f.write("*********************************************************************************************************************************"+'\n') 
    else:     
        best_model=model.state_dict()
        result_dir='results'
        os.makedirs(result_dir,exist_ok=True)
        base_dir='head_'+str(args.n_heads)+'_TPCA_'+str(args.num_TPCA)+'_w_cla_'+str(args.w_cla)+'_w_con_'+str(args.w_con)+'_lr_'+str(args.lr)
        os.makedirs(os.path.join(result_dir,base_dir),exist_ok=True)
        for epoch in range(start_epoch, args.epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)
            
            # train
            train_acc = train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

            # validation
            test_acc1, test_c_acc1, test_precision1, test_recall1, test_fscore1, test_bbox_miou1 = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device,name='endovis18')
            test_acc2, test_c_acc2, test_precision2, test_recall2, test_fscore2, test_bbox_miou2 = validate(args, val_loader=external_val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device,name='endovis17')
            test_acc3, test_c_acc3, test_precision3, test_recall3, test_fscore3, test_bbox_miou3 = validate(args, val_loader=SGH_val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device,name='SGH')
            test_avg_acc_miou1 = (test_acc1 + test_bbox_miou1) / 2                        
            
            if test_acc1 >= best_results[0]:
                epochs_since_improvement = 0
                
                best_results[0] = test_acc1
                best_epoch[0] = epoch
                best_acc_epoch_fscore = test_fscore1
                best_acc_epoch_miou = test_bbox_miou1
                best_acc_epoch_avg_acc_miou = test_avg_acc_miou1
                # checkpoint_path = os.path.join(result_dir,base_dir,'best_18.pth')
                # torch.save(model.state_dict(), checkpoint_path)  
                        
            else:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            if test_acc2 >= best_results[1]:
                epochs_since_improvement = 0
                
                best_results[1] = test_acc2
                best_epoch[1] = epoch
                # checkpoint_path = os.path.join(result_dir,base_dir,'best_17.pth')
                # torch.save(model.state_dict(), checkpoint_path)
            
            if test_acc3 >= best_results[2]:
                epochs_since_improvement = 0
                
                best_results[2] = test_acc3
                best_epoch[2] = epoch
                checkpoint_path = os.path.join(result_dir,base_dir,'best_SGH.pth')
                torch.save(model.state_dict(), checkpoint_path)
            
            print('Best Acc epoch 18:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_epoch[0], best_results[0], best_acc_epoch_fscore, best_acc_epoch_miou, best_acc_epoch_avg_acc_miou))
            print('Best Acc epoch 17:   %d | Acc: %.6f ' %(best_epoch[1], best_results[1]))
            print('Best Acc epoch SGH:   %d | Acc: %.6f ' %(best_epoch[2], best_results[2]))
            with open(os.path.join(result_dir,base_dir) + '.txt','a') as f:  
            
                f.write('Best Acc epoch 18:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f \n' %(best_epoch[0], best_results[0], best_acc_epoch_fscore, best_acc_epoch_miou, best_acc_epoch_avg_acc_miou)) 
                f.write('Best Acc epoch 17:   %d | Acc: %.6f \n' %(best_epoch[1], best_results[1]))
                f.write('Best Acc epoch SGH:   %d | Acc: %.6f \n' %(best_epoch[2], best_results[2]))                

            if test_bbox_miou1 >= best_miou:
                best_miou = test_bbox_miou1
                best_miou_epoch = epoch
                best_miou_epoch_acc = test_acc1
                best_miou_epoch_fscore = test_fscore1
                best_miou_epoch_avg_acc_miou = test_avg_acc_miou1
            
            print('Best mIoU epoch:  %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_miou_epoch, best_miou_epoch_acc, best_miou_epoch_fscore, best_miou, best_miou_epoch_avg_acc_miou))         
            with open(os.path.join(result_dir,base_dir) + '.txt','a') as f:    
                f.write('Best mIoU epoch:  %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f \n' %(best_miou_epoch, best_miou_epoch_acc, best_miou_epoch_fscore, best_miou, best_miou_epoch_avg_acc_miou))                 

            if test_avg_acc_miou1 >= best_avg_acc_miou:
                best_avg_acc_miou = test_avg_acc_miou1
                best_avg_acc_miou_epoch = epoch
                best_avg_epoch_acc = test_acc1
                best_avg_epoch_fscore = test_fscore1
                best_avg_epoch_miou = test_bbox_miou1

                  

            print('Best Avg epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f' %(best_avg_acc_miou_epoch, best_avg_epoch_acc, best_avg_epoch_fscore, best_avg_epoch_miou, best_avg_acc_miou))
            with open(os.path.join(result_dir,base_dir) + '.txt','a') as f:    
                f.write('Best Avg epoch:   %d | Acc: %.6f | FScore: %.6f | mIoU: %.6f | Avg: %.6f \n' %(best_avg_acc_miou_epoch, best_avg_epoch_acc, best_avg_epoch_fscore, best_avg_epoch_miou, best_avg_acc_miou))      
