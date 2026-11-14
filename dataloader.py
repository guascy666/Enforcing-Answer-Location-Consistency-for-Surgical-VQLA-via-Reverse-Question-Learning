'''
Description     : Dataloader for Localized-Answering task.
Paper           : CAT-ViL: Co-Attention Gated Vision-Language Embedding for Visual Question Localized-Answering in Robotic Surgery
Author          : Long Bai, Mobarakol Islam, Hongliang Ren
Lab             : Medical Mechatronics Lab, The Chinese University of Hong Kong
'''

import os
import glob
import h5py
from torchvision import models, transforms
import torch
from torch.utils.data import Dataset
from utils import *
import random
from PIL import Image
class FeatureExtractor(nn.Module):
    def __init__(self, patch_size = 4):
        super(FeatureExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(pretrained=True)
        self.img_feature_extractor = torch.nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))
        self.resize_dim = nn.AdaptiveAvgPool2d((patch_size,patch_size))
        
    def forward(self, img):
        outputs = self.resize_dim(self.img_feature_extractor(img))
        return outputs
    
'''
EndoVis18 VQLA dataloader
'''
class EndoVis18VQAClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqla/label/*.txt'
    	patch_size      = 5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4, robustpath=None,mode='train'):
             
        self.patch_size = patch_size
        self.foldder_head=folder_head
        self.mode=mode
        print("mode:",self.mode)
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        # self.robustpath=robustpath
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
        self.transform = transforms.Compose([transforms.Resize((300,256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ])
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        question = self.vqas[idx][1].split('|')[0]
        query=question
        #image_path = os.path.join('/media/cygzz/data/guanjl1/surgical_VQLA/datasets/EndoVis-18-VQLA_new/Images/seq_'+(loc[-3].split('_')[-1]),'seq_'+(loc[-3].split('_')[-1])+'_'+loc[-1].split('_')[0]+'.png')
        visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'.hdf5')  # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])

        
        
        bbox=list(self.vqas[idx][1].split('|')[2].replace(' ','').split(','))
        local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'.hdf5')
        local_data = h5py.File(local_path, 'r')    
        local_features = torch.from_numpy(local_data['visual_features'][:])
            
        # question and answer
        
        

        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        
        h=False
        v=False
        if self.mode=='train':

            label_item=str(self.vqas[idx][1].split('|')[1])
            

            if random.random() > 0.5:
                
                question = question.replace('left', '@').replace('right', 'left').replace('@', 'right')
                label_item = label_item.replace('left', '@').replace('right', 'left').replace('@', 'right')
                new1=1279-bbox_label_xyxy[2]
                new2=1279-bbox_label_xyxy[0]
                bbox_label_xyxy[0]=new1
                bbox_label_xyxy[2]=new2
                h=True



        
            if random.random() > 0.5:
                
                question = question.replace('top', '@').replace('bottom', 'top').replace('@', 'bottom')
                label_item = label_item.replace('top', '@').replace('bottom', 'top').replace('@', 'bottom')
                new3=1023-bbox_label_xyxy[3]
                new4=1023-bbox_label_xyxy[1]
                bbox_label_xyxy[1]=new3
                bbox_label_xyxy[3]=new4
                v=True

            if h==True and v==True:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_hv.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_hv.hdf5')
            
            if h==True and v==False:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_h.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_h.hdf5')
            
            if h==False and v==True:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_v.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_v.hdf5')


            assert bbox_label_xyxy[0] < bbox_label_xyxy[2], f"fuck"
            assert bbox_label_xyxy[1] < bbox_label_xyxy[3], f"fuck"

            bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
            bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])

            frame_data = h5py.File(visual_feature_loc, 'r')    
            visual_features = torch.from_numpy(frame_data['visual_features'][:])
            local_data = h5py.File(local_path, 'r')    
            local_features = torch.from_numpy(local_data['visual_features'][:])
            label=self.labels.index(label_item)

            return loc[-4] + '/' + loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize,local_features
        
        else:

            bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
            bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
            label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
            return loc[-3] + '/' + loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize


class EndoVis17VQAClassificationValidation(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4, robustpath=None):
             
        self.patch_size = patch_size
        self.folder_head=external_folder_head
        
        # files, question and answers
        filenames = []
        filenames =  glob.glob(external_folder_head + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))

        self.robustpath=robustpath
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = self.folder_head+ '/swin_features/'+ (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('.')[0]+'.hdf5'
        # visual_feature_loc = loc[0]+'/'+loc[1]+self.robustpath+str(self.patch_size)+'x'+str(self.patch_size)+'/'+loc[-1].split('.')[0]+'.hdf5'   # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
        return loc[-1].split('.')[0], visual_features, question, label, bbox_label_cxcywh_nomalize


class SGHVQAClassificationValidation(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4, robustpath=None):
             
        self.patch_size = patch_size
        self.folder_head=external_folder_head
        
        # files, question and answers
        filenames = []
        clip=['clip1','clip2','clip3','clip4','clip5','clip6','clip7','clip8','clip9','clip10']
        for curr_seq in clip: filenames = filenames + glob.glob(external_folder_head+'/' + str(curr_seq) + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))

        self.robustpath=robustpath
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = '/media/cygzz/data/guanjl1/surgical_VQLA/datasets/SGH_VQLA/swin_features/'+loc[-2]+'/'+ (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('.')[0]+'.hdf5'
        # visual_feature_loc = loc[0]+'/'+loc[1]+self.robustpath+str(self.patch_size)+'x'+str(self.patch_size)+'/'+loc[-1].split('.')[0]+'.hdf5'   # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]

        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([670, 536, 670, 536])
        return loc[-2]+'_'+loc[-1].split('.')[0], visual_features, question, label, bbox_label_cxcywh_nomalize




class EndoVis18VQAClassification_noise(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqla/label/*.txt'
    	patch_size      = 5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4, robustpath=None,mode='train',noise_type=None):
             
        self.patch_size = patch_size
        self.foldder_head=folder_head
        self.mode=mode
        print("mode:",self.mode)
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        # self.robustpath=robustpath
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
        self.transform = transforms.Compose([transforms.Resize((300,256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ])
        self.noise_type=noise_type
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        question = self.vqas[idx][1].split('|')[0]
        query=question
        #image_path = os.path.join('/media/cygzz/data/guanjl1/surgical_VQLA/datasets/EndoVis-18-VQLA_new/Images/seq_'+(loc[-3].split('_')[-1]),'seq_'+(loc[-3].split('_')[-1])+'_'+loc[-1].split('_')[0]+'.png')
        visual_feature_loc = os.path.join('/media/cygzz/data/guanjl1/surgical_VQLA/conrupt/18/swin_features',self.noise_type,loc[-3], (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-3]+'_'+loc[-1].split('_')[0]+'.hdf5')  # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])

        
        
        bbox=list(self.vqas[idx][1].split('|')[2].replace(' ','').split(','))
        local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'.hdf5')
        local_data = h5py.File(local_path, 'r')    
        local_features = torch.from_numpy(local_data['visual_features'][:])
            
        # question and answer
        
        

        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        
        h=False
        v=False
        if self.mode=='train':

            label_item=str(self.vqas[idx][1].split('|')[1])
            

            if random.random() > 0.5:
                
                question = question.replace('left', '@').replace('right', 'left').replace('@', 'right')
                label_item = label_item.replace('left', '@').replace('right', 'left').replace('@', 'right')
                new1=1279-bbox_label_xyxy[2]
                new2=1279-bbox_label_xyxy[0]
                bbox_label_xyxy[0]=new1
                bbox_label_xyxy[2]=new2
                h=True



        
            if random.random() > 0.5:
                
                question = question.replace('top', '@').replace('bottom', 'top').replace('@', 'bottom')
                label_item = label_item.replace('top', '@').replace('bottom', 'top').replace('@', 'bottom')
                new3=1023-bbox_label_xyxy[3]
                new4=1023-bbox_label_xyxy[1]
                bbox_label_xyxy[1]=new3
                bbox_label_xyxy[3]=new4
                v=True

            if h==True and v==True:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_hv.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_hv.hdf5')
            
            if h==True and v==False:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_h.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_h.hdf5')
            
            if h==False and v==True:
                visual_feature_loc = os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_features', (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_v.hdf5')
                local_path=os.path.join(self.foldder_head+(loc[-3].split('_')[-1]),'swin_local_features',(str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('_')[0]+'_'+query+'_v.hdf5')


            assert bbox_label_xyxy[0] < bbox_label_xyxy[2], f"fuck"
            assert bbox_label_xyxy[1] < bbox_label_xyxy[3], f"fuck"

            bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
            bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])

            frame_data = h5py.File(visual_feature_loc, 'r')    
            visual_features = torch.from_numpy(frame_data['visual_features'][:])
            local_data = h5py.File(local_path, 'r')    
            local_features = torch.from_numpy(local_data['visual_features'][:])
            label=self.labels.index(label_item)

            return loc[-4] + '/' + loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize,local_features
        
        else:

            bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
            bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
            label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
            return loc[-4] + '/' + loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize


class EndoVis17VQAClassificationValidation_noise(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4, robustpath=None,noise_type=None):
             
        self.patch_size = patch_size
        self.folder_head=external_folder_head
        
        # files, question and answers
        filenames = []
        filenames =  glob.glob(external_folder_head + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))

        self.robustpath=robustpath
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
        self.noise_type=noise_type
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
   
        visual_feature_loc = os.path.join('/media/cygzz/data/guanjl1/surgical_VQLA/conrupt/17/swin_features',self.noise_type, (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('.')[0]+'.hdf5') 
        # visual_feature_loc = self.folder_head+ '/swin_features/'+ (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('.')[0]+'.hdf5'
        # visual_feature_loc = loc[0]+'/'+loc[1]+self.robustpath+str(self.patch_size)+'x'+str(self.patch_size)+'/'+loc[-1].split('.')[0]+'.hdf5'   # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
        return loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize


class SGHVQAClassificationValidation_noise(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4, robustpath=None,noise_type=None):
             
        self.patch_size = patch_size
        self.folder_head=external_folder_head
        
        # files, question and answers
        filenames = []
        clip=['clip1','clip2','clip3','clip4','clip5','clip6','clip7','clip8','clip9','clip10']
        for curr_seq in clip: filenames = filenames + glob.glob(external_folder_head+'/' + str(curr_seq) + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))

        self.robustpath=robustpath
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom',
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 
                        'clip_applier', 'suction', 'stapler']
        self.noise_type=noise_type
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = os.path.join('/media/cygzz/data/guanjl1/surgical_VQLA/conrupt/SGH/swin_features',self.noise_type,loc[-2], (str(self.patch_size)+'x'+str(self.patch_size))+'_'+loc[-1].split('.')[0]+'.hdf5') 
        # visual_feature_loc = loc[0]+'/'+loc[1]+self.robustpath+str(self.patch_size)+'x'+str(self.patch_size)+'/'+loc[-1].split('.')[0]+'.hdf5'   # robustness
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]

        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([670, 536, 670, 536])
        return loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize
    

