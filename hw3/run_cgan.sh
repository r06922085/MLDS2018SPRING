mkdir -p hw3_2/model_file
wget https://www.dropbox.com/s/m8g8w4jlb327bu8/cgan.ckpt.data-00000-of-00001 -P hw3_2/model_file
wget https://www.dropbox.com/s/b9gjs5dh6qfnsjj/cgan.ckpt.index -P hw3_2/model_file
wget https://www.dropbox.com/s/fucrwc50x44ot2v/cgan.ckpt.meta -P hw3_2/model_file
python3 hw3_2/generate.py $1
