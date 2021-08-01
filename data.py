'''
Module :  data
Author:  Nasibullah (nasibullah104@gmail.com)
Details : This module creates datasets and dataloaders suitable for feeding data to models.
          It Currently supports MSCOCO2014. 
          
'''

import os
import json
import random
import itertools
from PIL import Image
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.nn import functional as F


from dictionary import Vocabulary,EOS_token,PAD_token,SOS_token,UNK_token


class COCO14Dataset(Dataset):
    def __init__(self,utils,coco,voc,transforms=None):
        self.coco = coco
        self.voc = voc
        self.transforms = transforms
        self.normalize = utils.normalizeString
    def __len__(self):
        return len(self.coco)
    def __getitem__(self,idx):
        img,target = self.coco[idx]
        ide = self.coco.ids[idx]
        lbl = self.normalize(random.choice(target))
        label = []
        for s in lbl.split(' '):
            if s in list(self.voc.word2index.keys()):
                label.append(self.voc.word2index[s])
            else:
                label.append(UNK_token)
        label = label +[EOS_token]
        
        return img, label,ide
    
class COCO2014Test(Dataset):
    
    def __init__(self,test_image_path,id2img,transform=None):
        self.image_path = test_image_path
        self.id2fname = id2img
        self.idlist = list(id2img.keys())
        self.transform = transform
        
    def __len__(self):
        return len(self.id2fname)
    def __getitem__(self,idx):
        ide = self.idlist[idx]
        img = Image.open(os.path.join(self.image_path,self.id2fname[ide])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return ide,img
    
def collate_fn(batch):
    data = [item[0] for item in batch]
    images = torch.stack(data,0)
    
    ides = torch.tensor([item[2] for item in batch])
    
    label = [item[1] for item in batch]
    max_target_len = max([len(indexes) for indexes in label])
    padList = list(itertools.zip_longest(*label, fillvalue = 0))
    lengths = torch.tensor([len(p) for p in label])
    padVar = torch.LongTensor(padList)
    
    m = []
    for i, seq in enumerate(padVar):
        #m.append([])
        tmp = []
        for token in seq:
            if token == 0:
                tmp.append(int(0))
            else:
                tmp.append(1)
        m.append(tmp)
    m = torch.tensor(m)
    
    return images, padVar, m, max_target_len, ides

class DataHandler:
    
    def __init__(self,cfg,path,voc):
        self.cfg = cfg
        self.voc = voc
        self.path = path
        self.data_transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

        self.coco_train = dset.CocoCaptions(root=self.path.train_image_path,annFile=self.path.train_annotation_file,transform=self.data_transform)
        self.coco_val = dset.CocoCaptions(root=self.path.val_image_path,annFile=self.path.val_annotation_file,transform=self.data_transform) 

        self.id2fname = {}
        test_info = json.load(open(self.path.test_info_path))
        for img in test_info['images']:
            self.id2fname[img['id']] = img['file_name']

    def getDataSets(self,utils):     
        train_dset = COCO14Dataset(utils,self.coco_train,self.voc,transforms=self.data_transform)
        val_dset = COCO14Dataset(utils,self.coco_val,self.voc)
        test_dset = COCO2014Test(self.path.test_image_path,self.id2fname,self.data_transform)
        
        return train_dset, val_dset, test_dset
    def getDataLoaders(self,train_dset,val_dset,test_dset):

        train_loader=DataLoader(train_dset,batch_size = self.cfg.batch_size, num_workers = 8,shuffle = True,
                            collate_fn = collate_fn, drop_last=True)

        val_loader = DataLoader(val_dset,batch_size = self.cfg.val_batch_size, num_workers = 8,shuffle = False,collate_fn = collate_fn,drop_last=False)
        test_loader = DataLoader(test_dset,batch_size = 35, num_workers = 8,shuffle = False,drop_last=False)
        
        return train_loader,val_loader,test_loader
