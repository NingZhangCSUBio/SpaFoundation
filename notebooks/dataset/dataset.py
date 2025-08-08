import torch
import numpy as np
from torchvision.transforms import functional as TF
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent 

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
    
class ViT_SKIN(torch.utils.data.Dataset):
    "patient-level cross validation"
    def __init__(self,train=True,gene_list=None,fold=0,test_slice=0,patch_size=224,gene_tag='HEG'):
        super(ViT_SKIN, self).__init__()
        self.dir = str(BASE_DIR)+'/cscc/GSE144240_RAW/'
        self.r = patch_size//2
        self.choosegene = gene_tag#HEG,HVG

        patients = ['P2', 'P5', 'P9', 'P10']
        reps = ['rep1', 'rep2', 'rep3']
        names = []
        for i in patients:
            for j in reps:
                names.append(i+'_ST_'+j)

        if self.choosegene=='HEG':
            gene_list = list(np.load(str(BASE_DIR)+'/cscc/genes_cscc_heg.npy',allow_pickle=True))
        elif self.choosegene=='HVG':
            gene_list = list(np.load(str(BASE_DIR)+'/cscc/genes_cscc_hvg.npy',allow_pickle=True))
        else:
            print('The gene list is wrong!')
        self.gene_list = gene_list
        self.train = train

        samples = names
        te_names = samples[fold*3:(fold+1)*3]
        tr_names=[item for item in samples if item not in te_names]

        if train:
            names = tr_names
            print('Training the slides:',names)
        else:
            names = te_names
            names = [names[test_slice]]
            print('Testing the slides:',names)
            
        self.exp_list=[] 
        self.center_list=[]
        self.loc_list=[] 
        self.img_list=[]
        self.ll_list=[]
        print('Loading data...')
        self.gene_set = list(gene_list)
        for i in names:
            meta_info=self.get_meta(i)
            exp_info=scp.transform.log(scp.normalize.library_size_normalize(meta_info[self.gene_set].values))
            center_info=np.floor(meta_info[['pixel_x','pixel_y']].values).astype(int)
            loc_info=meta_info[['x','y']].values
            self.exp_list.append(exp_info)
            self.center_list.append(center_info)
            self.loc_list.append(loc_info)
            
            img_info=np.array(self.get_img(i))#H,W,C
            
            
            for x,y in center_info:
                patch = img_info[(y-self.r):(y+self.r),(x-self.r):(x+self.r),:]#H,W,C
                self.img_list.append(patch)
        self.exp_list=np.concatenate(self.exp_list, axis=0)
        self.center_list=np.concatenate(self.center_list, axis=0)
        self.loc_list=np.concatenate(self.loc_list, axis=0)
        print(self.exp_list.shape,self.center_list.shape,self.center_list.shape)
        
    def transform(self, image):
        image = Image.fromarray(image)
        
        if self.train:    
            # Random flipping and rotations
            if random.random() > 0.5:
                image = TF.hflip(image)
            if random.random() > 0.5:
                image = TF.vflip(image)
            
            
            angle = random.choice([180, 90, 0, -90])
            image = TF.rotate(image, angle)
            
        # Convert to tensor
        image = TF.to_tensor(image)
        
        return image

    def __getitem__(self, index):

        patch=self.img_list[index]
        patch=self.transform(patch)
        center=self.center_list[index]
        position= self.loc_list[index]
        exp=self.exp_list[index]
        exp = torch.Tensor(exp)
        position=torch.LongTensor(position)

        if self.train:
            return patch, exp
        else: 
            return patch, position, exp, torch.Tensor(center)
        
    def __len__(self):
        return len(self.exp_list)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im
    
    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df
    

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta

class SKIN_VIS(torch.utils.data.Dataset):
    'for visualization'
    def __init__(self,patient_name='P2',slice_index='rep2'):
        super(SKIN_VIS, self).__init__()
        
        self.dir = str(BASE_DIR)+'/cscc/GSE144240_RAW/'
        self.r=112

        gene_list = list(np.load(str(BASE_DIR)+'/cscc/genes_cscc_heg.npy',allow_pickle=True))
        self.gene_list = gene_list

        names=patient_name+'_ST_'+slice_index
        self.exp_list=[] 
        self.center_list=[]
        self.loc_list=[] 
        self.img_list=[]
        self.ll_list=[]
        print('Loading data...')
        self.gene_set = list(gene_list)

        meta_info=self.get_meta(names)
        exp_info=scp.transform.log(scp.normalize.library_size_normalize(meta_info[self.gene_set].values))
        center_info=np.floor(meta_info[['pixel_x','pixel_y']].values).astype(int)
        loc_info=meta_info[['x','y']].values
        self.exp_list.append(exp_info)
        self.center_list.append(center_info)
        self.loc_list.append(loc_info)
        
        img_info=np.array(self.get_img(names))#H,W,C
        
        for x,y in center_info:
            patch = img_info[(y-self.r):(y+self.r),(x-self.r):(x+self.r),:]#H,W,C
            self.img_list.append(patch)
        self.exp_list=np.concatenate(self.exp_list, axis=0)
        self.center_list=np.concatenate(self.center_list, axis=0)
        self.loc_list=np.concatenate(self.loc_list, axis=0)
        print(self.exp_list.shape,self.center_list.shape,self.center_list.shape)
        
    def transform(self, image):
        image = Image.fromarray(image)
        
        # Convert to tensor
        image = TF.to_tensor(image)
        
        return image

    def __getitem__(self, index):

        patch=self.img_list[index]
        patch=self.transform(patch)
        center=self.center_list[index]
        position= self.loc_list[index]
        exp=self.exp_list[index]
        exp = torch.Tensor(exp)
        position=torch.LongTensor(position)

        return patch, position, exp, torch.Tensor(center)
        
    def __len__(self):
        return len(self.exp_list)

    def get_img(self,name):
        path = glob.glob(self.dir+'*'+name+'.jpg')[0]
        im = Image.open(path)
        return im
    
    def get_cnt(self,name):
        path = glob.glob(self.dir+'*'+name+'_stdata.tsv')[0]
        df = pd.read_csv(path,sep='\t',index_col=0)
        return df
    

    def get_pos(self,name):
        path = glob.glob(self.dir+'*spot*'+name+'.tsv')[0]
        df = pd.read_csv(path,sep='\t')

        x = df['x'].values
        y = df['y'].values
        x = np.around(x).astype(int)
        y = np.around(y).astype(int)
        id = []
        for i in range(len(x)):
            id.append(str(x[i])+'x'+str(y[i])) 
        df['id'] = id

        return df

    def get_meta(self,name,gene_list=None):
        cnt = self.get_cnt(name)
        pos = self.get_pos(name)
        meta = cnt.join(pos.set_index('id'),how='inner')

        return meta
