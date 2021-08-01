'''
Module :  dictionary
Author:  Nasibullah (nasibullah104@gmail.com)
          
'''


try:
    import pickle5 as pickle
except:
    import pickle

SOS_token = 1
EOS_token = 2
PAD_token = 0
UNK_token = 3

class Vocabulary:
    
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD":PAD_token,"EOS":EOS_token,"SOS":SOS_token, "UNK":UNK_token}
        self.word2count = {}
        self.index2word = {PAD_token:"PAD",EOS_token:"EOS",SOS_token:"SOS", UNK_token:"UNK"}
        self.num_words = 4
        
    def addSentence(self,sentence): #Add Sentence to vocabulary
        for word in sentence.split(' '):
            self.addWord(word)
            
    def addWord(self, word):  # Add words to vocabulary
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            if self.trimmed == False:
                self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            if self.trimmed == False:
                self.word2count[word] += 1
            
    def save(self,word2index_dic = 'word2index_dic', index2word_dic = 'index2word_dic',
         word2count_dic = 'word2count_dic'):

        with open('Save/'+word2index_dic+'.p', 'wb') as fp:
            pickle.dump(self.word2index, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Save/'+index2word_dic+'.p', 'wb') as fp:
            pickle.dump(self.index2word, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open('Save/'+word2count_dic+'.p', 'wb') as fp:
            pickle.dump(self.word2count, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, word2index_dic = 'word2index_dic', index2word_dic = 'index2word_dic',
             word2count_dic = 'word2count_dic'):
        
        with open('Save/'+word2index_dic+'.p', 'rb') as fp:
            self.word2index = pickle.load(fp)
            
        with open('Save/'+index2word_dic+'.p', 'rb') as fp:
            self.index2word = pickle.load(fp)
            
        with open('Save/'+word2count_dic+'.p', 'rb') as fp:
            self.word2count = pickle.load(fp)
            
        self.num_words = len(self.word2index)
        
    def trim(self, min_count):  # Trim Rare words with frequency less than min_count
        if self.trimmed:
            print('Already trimmed before')
            return 0
        self.trimmed = True
        
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {"PAD":PAD_token,"EOS":EOS_token,"SOS":SOS_token}
        #self.word2count = {}
        self.index2word = {PAD_token:"PAD",EOS_token:"EOS",SOS_token:"SOS"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)
            if word not in self.word2count:
                del self.word2count[word]

