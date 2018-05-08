
# coding: utf-8

# In[1]:


from collections import Counter
import numpy as np

class Dictionary():
    train_dict = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
    voc_size = 4
    
    def LoadData(self, file_name):
        # load file
        with open(file_name, 'r', encoding='utf8') as f:
            train_words = f.read().replace(' ','')
            train_words = Counter(train_words).most_common(3000)
        return train_words
    
    def BuildDict(self):
        words = self.LoadData()
        for word in words:
            self.train_dict.append(word[0])
            self.voc_size += 1

    def word2index(self, word):
        index = 3
        try:
            index = self.train_dict.index(word)
        except:
            pass
        return index
    
    def sentence2index(self, sentence, max_len=25):
        words = list(sentence)
        index = []
        for word in words:
            if len(index) < max_len-1:
                index.append(self.word2index(word))
            else:
                break
        index.append(self.word2index('<EOS>'))
        while len(index) < max_len:
            index.append(self.word2index('<PAD>'))
        return index
    
    def index2sentence(self, index_list):
        sentence = ''
        for index in index_list:
            sentence += self.train_dict[index]
        sentence = sentence.replace('<PAD>','')
        sentence = sentence.replace('<BOS>','')
        sentence = sentence.replace('<EOS>','')
        sentence = sentence.replace('<UNK>','')
        return sentence

# In[2]:


class DataManager():
    data = []
    label = []
    train_dict = None
    def __init__(self):
        self.train_dict = Dictionary()
        
    def LoadData(self, file_name, max_len = 25):
        # load file
        with open(file_name, 'r', encoding='utf8') as f:
            sentences = f.read().replace(' ','').split('\n')
            
        for i in range(len(sentences)-1):
            if sentences[i+1] == '+++$+++':
                continue
            self.data.append(self.train_dict.sentence2index(sentences[i],max_len))
            self.label.append(self.train_dict.sentence2index(sentences[i+1],max_len))
        label_bos = np.ones((len(self.label),1))
        self.label = np.hstack((label_bos,self.label))
        
    def LoadTestData(self, file_name, max_len=25):
        # load file
        with open(file_name, 'r', encoding='utf8') as f:
            sentences = f.read().replace(' ','').split('\n')
        for i in range(len(sentences)-1):
            self.data.append(self.train_dict.sentence2index(sentences[i],max_len))