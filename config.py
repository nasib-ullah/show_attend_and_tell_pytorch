'''
Module :  config
Author:  Nasibullah (nasibullah104@gmail.com)
Details : Ths module consists of all hyperparameters and path details.
          Only changing this module is enough to play with different model configurations. 
          
'''


import torch
import os

class Config:
    '''
    Hyperparameter settings for Show attend and Tell
    '''
    def __init__(self,model_name = 'SAT'):
        
        self.model_name = model_name
        self.cuda_device_id = 1
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(self.cuda_device_id)) 
        else:
            self.device = torch.device('cpu')
            
            
        #data configuration
        self.batch_size = 64 
        self.val_batch_size = 61
        
        #encoder configuration
        self.encoder_arch = 'vgg'; assert self.encoder_arch in ['vgg','resnet']
        self.feat_size = 512 # encoder's annotation vector length
        self.feat_len = 196  # (196=14*14) output is taken from intermediate convolutional layer of encoder
        
        #decoder configuration
        self.embedding_size = 512  # word embedding size
        self.hidden_size = 512    # Hidden state vector size of decoder LSTM
        self.decoder_input_size = self.embedding_size + self.feat_size
        self.attn_size = 256     #bottleneck size for attention module
        self.rnn_dropout = 0.5   # Dropout probability for decoder LSTM layer
        self.num_layers = 1 
        self.num_directions = 1
        
        #Training configuration
        self.teacher_forcing_ratio = 0.7 
        self.clip = 5 # clip the gradient to counter exploding gradient problem
        self.encoder_lr = 1e-5
        self.decoder_lr = 1e-3
        self.print_every = 400
        
    def update(self):
        self.decoder_input_size = self.embedding_size+self.feat_size

class Path:
    '''
    Currently supports MSCOCO2014
    '''
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
        self.prediction_path = 'results'
        self.saved_models_path = 'Save'
        
        self.train_image_path = os.path.join(self.dataset_path,'train2014')
        self.val_image_path = os.path.join(self.dataset_path + 'val2014')
        self.test_image_path = os.path.join(self.dataset_path + 'test2014')

        self.annotation_path = os.path.join(dataset_path + 'annotations')
        self.train_annotation_file = os.path.join(self.annotation_path,'captions_train2014.json')
        self.val_annotation_file = os.path.join(self.annotation_path,'captions_val2014.json')

        self.prediction_filepath= 'results'
        self.test_info_path = os.path.join(self.annotation_path,'image_info_test2014.json')
