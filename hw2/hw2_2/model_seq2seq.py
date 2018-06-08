# coding: utf-8

from utils.dataManager import DataManager
from utils.Seq2Seq import Seq2Seq
import tensorflow as tf
import numpy as np
import os
import sys
import pickle
import time

def main():
    start_time = time.time()
    dataset = DataManager()

    # load dictionary
    dict_file = open('data/dictionary.txt','rb')
    word_list = pickle.load(dict_file)
    voc_size = len(word_list)
    dataset.train_dict.train_dict = word_list
    dataset.train_dict.voc_size = voc_size
    dictionary = dataset.train_dict

    # load data from input file
    dataset.LoadTestData(sys.argv[1])
    test_data=np.asarray(dataset.data)
   
    batch_size = 100
    tf.reset_default_graph()
    model = Seq2Seq(voc_size,batch_size=batch_size,mode='test')
    model.compile()
    output_file = open(sys.argv[2],'w',encoding='utf8')
    count = 0 
    for i in range(int(len(test_data)/batch_size)):
        predict_labels = model.predict(test_data[i*batch_size:((i+1)*batch_size)])
        for j in range(batch_size):
            count += 1
            result = dictionary.index2sentence(predict_labels[j][0])
            if result.replace(' ','') == '':
                result = '...'
            output_file.write("%s\n"%result)
    while count <= len(test_data):
        count += 1
        output_file.write("...\n")
    print('Cost time: %.2f minutes'%((time.time()-start_time)/60.0))
    
    
    # greedy decoder
    '''
    batch_size = 100
    tf.reset_default_graph()
    model = Seq2Seq(voc_size,batch_size=batch_size,mode='test',beam=False)
    model.compile()
    output_file = open(sys.argv[2],'w',encoding='utf8')
    count = 0
    for i in range(int(len(test_data)/batch_size)):
        predict_labels = model.predict(test_data[i*batch_size:((i+1)*batch_size)])
        for j in range(batch_size):
            count += 1
            result = dictionary.index2sentence(predict_labels[j])
            if result.replace(' ','') == '':
                result = '...'
            output_file.write("%s\n"%result)
    while count < len(test_data):
        count += 1
        output_file.write("...\n")
    print('Cost time: %.2f minutes'%((time.time()-start_time)/60.0))
    '''
        
if __name__ == '__main__':
    main()