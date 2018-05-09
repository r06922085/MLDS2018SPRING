#!/bin/bash
mkdir -p train_model_file
mkdir -p test_model_file
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_2/test_model_file/chatbot.ckpt.data-00000-of-00001 -P test_model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_2/test_model_file/chatbot.ckpt.index -P test_model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_2/test_model_file/chatbot.ckpt.meta -P test_model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_2/test_model_file/checkpoint -P test_model_file/
python3 model_seq2seq.py $1 $2