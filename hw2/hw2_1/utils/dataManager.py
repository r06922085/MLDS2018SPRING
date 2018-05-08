import numpy as np
import json

PAD = '<PAD>' # index 0
BOS = '<BOS>' # index 1
EOS = '<EOS>' # index 2
UNK = '<UNK>' # index 3

class Dictionary():
    word_list = [PAD, BOS, EOS, UNK]
    voc_size = 4
    
    def __init__(self, words = []):
        self.add_words(words)
    
    def add_words(self, labels):
        words = [word for label in labels for word_list in label for word in word_list] # flatten labels data to word list
        for word in words:
            if word not in self.word_list:
                self.word_list.append(word)
        self.voc_size = len(self.word_list)

    def word2index(self, word):
        index = 3 # 3 is the index of <UNK>
        try:
            index = self.word_list.index(word)
        except:
            pass
        return index
    
    def index2word(self, index):
        return self.word_list[index]
    
    def convert(self, index_or_word):
        """
        Convert between word and index
        """
        if type(index_or_word) is str:
            return self.word2index(index_or_word)
        return self.index2word(index_or_word)
    
    def wordlist2index(self, word_list, max_len):
        result = []
        for word in word_list:
            result.append(self.word2index(word))
        if max_len is not None:
            while len(result) < max_len:
                PAD_index = self.word2index('<PAD>')
                result.append(PAD_index)
        return result
    
    def indexlist2wordlist(self, index_list):
        result = []
        for index in index_list:
            result.append(self.index2word(index))
        return result
    
    def to_one_hot(self, index_or_word):
        """
        Word or index to one hot format
        """
        index = -1
        input_type = type(index_or_word)
        if input_type is str:
            index = self.word2index(index_or_word)
        else:
            index = index_or_word
        one_hot = np.zeros(self.voc_size)
        one_hot[index] = 1
        return one_hot
        
class DataManager():
    clean_train_dict = Dictionary()
    raw_data = None
    train_data = None
    test_data = None
    train_label = None
    test_label = None
    clean_train_label = None
    clean_test_label = None
    
    # data used for training
    train_x = None
    test_x = None
    train_y = None
    test_y = None
    train_y_first = None
    test_y_first = None
    
    def __init__(self, path = 'MLDS_hw2_1_data/', load_feature = True):
        self.LoadData(path, load_feature)
        self.BuildCleanLabel()
        self.BuildCleanTrainDict()
        
    def LoadData(self, path = 'MLDS_hw2_1_data/', load_feature = True):
        """ Load and return all data and labels from TA's files
        Args:
            path : directory of TA's files
            load_feature : boolean of whether load from TA's feature or not(raw video)
        Load data as in raw_train_dataset and raw_test_dataset
        """

        if not load_feature:
            print('Not support load raw video yet. Please set "load_feature" True.')
            return

        # id, label, data
        train_id = None
        test_id = None
        train_label = None
        test_label = None
        train_data = []
        test_data = []

        # load file id
        with open(path + 'training_id.txt', 'r') as f:
            train_id = f.read().split('\n')[:-1]
        with open(path + 'testing_id.txt', 'r') as f:
            test_id = f.read().split('\n')[:-1]
        # load labels
        with open(path + 'training_label.json', 'r') as f:
            train_label = json.load(f)
        with open(path + 'testing_label.json', 'r') as f:
            test_label = json.load(f)
        # load data
        data_folder = ['video/', 'feat/']
        train_data_path = path + 'training_data/' + data_folder[int(load_feature)]
        test_data_path = path + 'testing_data/' + data_folder[int(load_feature)]

        ## load training data
        for file_id in train_id:
            filename = train_data_path + file_id
            if load_feature:
                file = np.load(filename + '.npy')
            """
            else:
                load avi file here
            """
            train_data.append(file)

        ## load testing data
        for file_id in test_id:
            filename = test_data_path + file_id
            if load_feature:
                file = np.load(filename + '.npy')
            """
            else:
                load avi file here
            """
            test_data.append(file)
        dataset = {'train_id':train_id, 'test_id':test_id, 'train_data':train_data,
                   'test_data':test_data, 'train_label':train_label, 'test_label':test_label}
        self.raw_data = dataset
        
        # split label sentence to word list
        train_label = [[sentence.split() for sentence in label['caption']] for label in train_label]
        test_label = [[sentence.split() for sentence in label['caption']] for label in test_label]
            
        self.train_data = train_data
        self.test_data = test_data
        self.train_label = train_label
        self.test_label = test_label
    
    # data cleaning
    def BuildCleanLabel(self, clean_unk = True):
        """
        args:
            mode : 'train' or 'test', choose clean train or test dataset
            clean_unk : whether delete <UNK> word

        if the word is a special word, replace with unknown word,
        else modify it to more general form and replace it.
        save clean data in clean_train_dataset and clean_test_dataset
        """
        label_list = [self.train_label, self.test_label]
        clean_label_list = []
        for i, label in enumerate(label_list):
            clean_label = []
            for j, data in enumerate(label):
                tmp_data = []
                for k, wordlist in enumerate(data):
                    tmp_wl = []
                    for l, word in enumerate(wordlist):
                        tmp_w = word.strip('. ')
                        tmp_w = tmp_w.replace(',', '')
                        tmp_w = tmp_w.replace('"', '')
                        tmp_w = tmp_w.replace('!', '')
                        tmp_w = tmp_w.replace("`s", "'s")
                        tmp_w = tmp_w.replace("'a", "'s")
                        tmp_w = tmp_w.lower()

                        # check special word
                        for c in tmp_w:
                            if not ('a' <= c <='z' or 'A' <= c <='Z' or c in "'-&"+'"' or '0' <= c <= '9'):
                                tmp_w = UNK
                                break
                        tmp_wl.append(tmp_w)
                    if clean_unk:
                        try:
                            tmp_wl.remove(UNK)
                        except:
                            pass
                    tmp_data.append(tmp_wl)
                clean_label.append(tmp_data)
            clean_label_list.append(clean_label)
        self.clean_train_label = clean_label_list[0]
        self.clean_test_label = clean_label_list[1]
        
    def BuildCleanTrainDict(self, dtype = np.float32):
        """
        Input a word dictionary and a dataset,
        Use dataset to build word dictionary.
        """
        self.clean_train_dict.add_words(self.clean_train_label)
        
    def BuildTrainableData(self, max_len = None):
        self.train_x = self.train_data
        self.test_x = self.test_data
        train_test_label_set = [self.clean_train_label, self.clean_test_label]
        tmp_train_test_label = []
        tmp_train_test_first_label = []
        for i, label_set in enumerate(train_test_label_set):
            tmp_label_set = []
            tmp_label_first_set = []
            for j, label in enumerate(label_set):
                tmp_label = []
                for k, caption in enumerate(label):
                    caption_index = self.clean_train_dict.wordlist2index(caption, max_len)
                    tmp_label.append(caption_index)
                    if k == 0:
                        tmp_label_first_set.append(caption_index)
                tmp_label_set.append(tmp_label)
            tmp_train_test_label.append(tmp_label_set)
            tmp_train_test_first_label.append(tmp_label_first_set)
        self.train_y = tmp_train_test_label[0]
        self.test_y = tmp_train_test_label[1]
        self.train_y_first = tmp_train_test_first_label[0]
        self.test_y_first = tmp_train_test_first_label[1]
        return