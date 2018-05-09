mkdir model_file
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_1/model_file/attention.ckpt.data-00000-of-00001 -P model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_1/model_file/attention.ckpt.index -P model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_1/model_file/attention.ckpt.meta -P model_file/
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw2/hw2_1/model_file/checkpoint -P model_file/
python3 test.py $1 $2
