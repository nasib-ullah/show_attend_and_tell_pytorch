import torch
import os

class Config:
    if torch.cuda.is_available():
        device = torch.device('cuda') # In case of multiple GPU system use 'cuda:x' where x is GPU device id
    else:
        device = torch.device('cpu')
        
    batch_size = 64 #suitable for 11GB GPU with vgg as encoder and single layer LSTM as decoder.(approx 10.4 GB in use)
    val_batch_size = 61
    feat_size = 512 # encoder's annotation vector length
    feat_len = 196  # (196=14*14) output is taken from intermediate convolutional layer of encoder
    embedding_size = 512 # word embedding size
    hidden_size = 512 # Hidden state vector size of decoder LSTM
    attn_size = 256 #bottleneck size for attention module
    rnn_dropout = 0.5 # Dropout probability for decoder LSTM layer
    teacher_forcing_ratio = .5 #
    clip = 5 # clip the gradient to counter exploding gradient problem
    encoder_lr = 1e-3
    decoder_lr = 1e-5
    print_every = 400

class Path:
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

        self.prediction_file_path= os.path.join(self.prediction_path,'Predicted_Results')
        self.test_info_path = os.path.join(self.annotation_path,'image_info_test2014.json')
