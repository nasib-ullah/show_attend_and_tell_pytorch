'''
Module :  utils
Author:  Nasibullah (nasibullah104@gmail.com)
          
'''

import sys
import os
import re
import random
import unicodedata
import numpy as np
import torch


class Utils:
    '''
    Generic utility function class.
    '''
    @staticmethod
    def set_seed(seed):
        '''
          For reproducibility 
        '''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def unicodeToAscii(s):
        '''
        Turn a Unicode string to plain ASCII, 
        Thanks to https://stackoverflow.com/a/518232/2809427
        '''
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    @staticmethod
    def normalizeString(s):
        '''
        Lowercase, trim, and remove non-letter 
        '''
        s = s.lower().strip()
        s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    
    @staticmethod
    def target_tensor_to_caption(voc,target):
        '''
        Convert target tensor to Caption
        '''
        gnd_trh = []
        lend = target.size()[1]
        for i in range(lend):
            tmp = ' '.join(voc.index2word[x.item()] for x in targets[:,i])
            gnd_trh.append(tmp)
        return gnd_trh

    
    @staticmethod
    def maskNLLLoss(inp, target, mask, device):
        '''
        Masked cross-entropy loss calculation
        '''
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp.squeeze(0), 1, target.view(-1, 1)).squeeze(1).float())
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(device)
        return loss, nTotal.item()
