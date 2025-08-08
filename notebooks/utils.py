import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import random
import torch
import warnings
from scipy.stats import pearsonr
warnings.filterwarnings('ignore')


def seed_torch(seed=2024):
    #随机种子固定
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True



def check_and_create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"The folder {folder_path} not found, Creating it now......")
    else:
        print(f"The folder {folder_path} has been founded!")

def get_pltrainer(trainepochs,device_num,tag,fold):
    checkpoint_callback = ModelCheckpoint(
        dirpath='results/'+tag,  
        filename='fold_'+str(fold)+'_train_final_model',
        save_top_k=0, 
        every_n_epochs=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        logger=False,  
        max_epochs=trainepochs,
        callbacks=[checkpoint_callback],
        gpus=[device_num], 
    )
    
    return  trainer


def model_predict(model, test_loader, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')): 
    'infer test data in model'
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp,centers in test_loader:

            patch = patch.to(device)
            
            pred = model(patch)
            if preds is None:
                preds = pred.squeeze()
                gt = exp.squeeze()
            else:
                if pred.shape[0]!=1:
                    pred = pred.squeeze()
                    exp = exp.squeeze()
                preds = torch.cat((preds,pred),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
    preds = preds.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    return preds,gt

#predict for visualization
def model_predict_vis(model, test_loader, adata=None, attention=True, device = torch.device('cpu')): 
    model.eval()
    model = model.to(device)
    preds = None
    with torch.no_grad():
        for patch, position, exp, center in test_loader:

            patch, position = patch.to(device), position.to(device)
            
            pred = model(patch)

            if preds is None:
                preds = pred.squeeze()
                ct = center
                gt = exp.squeeze()
            else:
                if pred.shape[0]!=1:
                    pred = pred.squeeze()
                    exp = exp.squeeze()
                preds = torch.cat((preds,pred),dim=0)
                ct = torch.cat((ct,center),dim=0)
                gt = torch.cat((gt,exp),dim=0)
                
    preds = preds.cpu().squeeze().numpy()
    ct = ct.cpu().squeeze().numpy()
    gt = gt.cpu().squeeze().numpy()
    print(preds.shape)

    return preds, gt, ct

def performance(pred, true):

    corr2 = np.zeros(pred.shape[1])
    p_values2 = np.zeros(pred.shape[1])  # store the p-values of every gene
    for j in range(pred.shape[1]):
        corr2[j], p_values2[j] = pearsonr(pred[:, j], true[:, j])
    corr2 = corr2[~np.isnan(corr2)]
    p_values2 = p_values2[~np.isnan(p_values2)]

    medpcc_gene = np.median(corr2)
    print("Median correlation across genes (PCC): ", medpcc_gene)
    