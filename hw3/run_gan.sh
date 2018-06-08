mkdir -p hw3_1/model_file
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw3/WGAN_v2.ckpt.data-00000-of-00001 -P hw3_1/model_file
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw3/WGAN_v2.ckpt.index -P hw3_1/model_file
wget https://gitlab.com/bear987978897/MLDS/raw/master/hw3/WGAN_v2.ckpt.meta -P hw3_1/model_file
python3 hw3_1/generate.py $1
