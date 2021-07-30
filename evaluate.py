import os
import sys
import torch
import json
sys.path.append('coco_eval/pycocoevalcap/')
sys.path.append('coco_eval/pycocoevalcap/bleu')
sys.path.append('coco_eval/pycocoevalcap/cider')
from coco_eval.pycocotools.coco import COCO
from coco_eval.pycocoevalcap.eval import COCOEvalCap
import pickle
from config import Config


class Evaluator:
    
    def __init__(self,model_name,prediction_filepath,reference_file,dataloader):
        self.arch_name = model_name
        self.prediction_filepath = prediction_filepath
        self.dataloader = dataloader
        self.coco = COCO(reference_file)
        self.scores = {}
        self.bleu4 = 0.206 # save best model based on bleu score
    
    def prediction_list(self,model):
        result = []
        ide_list = []
        caption_list =[]
        model.eval()
        with torch.no_grad():
            for data in self.dataloader:
                features, targets, mask, max_length,ides= data
                cap,cap_txt,_ = model.Greedy_Decoding(features.to(Config.device))
                ide_list += list(ides.cpu().numpy())
                caption_list += cap_txt
        for a in zip(ide_list,caption_list):
            result.append({'image_id':a[0].item(),'caption':a[1].strip()})      
        return result
    
    def prediction_file_generation(self,result,prediction_filename):
    
        self.predicted_file = os.path.join(self.prediction_filepath,prediction_filename) 
        with open(self.predicted_file, 'w') as fp:
            json.dump(result,fp)
            
    def evaluate(self,model,epoch):
        prediction_filename = self.arch_name+str(epoch)+'.json'
        result = self.prediction_list(model)
        self.prediction_file_generation(result,prediction_filename)
        
        cocoRes = self.coco.loadRes(self.predicted_file)
        cocoEval = COCOEvalCap(self.coco,cocoRes)
        scores = cocoEval.evaluate()
        self.scores[epoch] = scores
        if scores[0][1][3] > self.bleu4:
            self.bleu4 = scores[0][1][3] 
            self.save_model(model,epoch)
        return scores
    def save_model(self,model,epoch):
        print('Better result saving models....')
        encoder_filename = 'Save/'+ Config.model_name+'encoder_'+str(epoch)+'.pt'
        decoder_filename = 'Save/'+ Config.model_name+'decoder_'+str(epoch)+'.pt'
        torch.save(model.encoder.state_dict(),encoder_filename)
        torch.save(model.decoder.state_dict(),decoder_filename)
        
