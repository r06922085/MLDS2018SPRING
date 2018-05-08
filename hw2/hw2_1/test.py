from utils.dataManager import DataManager
from models.hw2_1_S2VT_attention import Seq2Seq as S2VT_attention
import tensorflow as tf
import sys

data_dir = sys.argv[1]
output_dir = sys.argv[2]

def main():
    max_len = 40
    dataset = DataManager(path = data_dir)
    dictionary = dataset.clean_train_dict
    voc_size = dictionary.voc_size
    dataset.BuildTrainableData(max_len = max_len)

    tf.reset_default_graph()
    model2 = S2VT_attention(voc_size = voc_size, max_len = max_len, dtype = tf.float32)
    model2.compile()
    model2.restore()
    test_id = dataset.raw_data['test_id']

    with open(output_dir, 'w') as f:
        for i, vid in enumerate(test_id):
            predict = model2.predict(dataset.test_x[i])
            predict = dataset.clean_train_dict.indexlist2wordlist(predict)
            sentence = ''
            for word in predict:
                if word == '<PAD>':
                    sentence = sentence.strip()
                    break
                sentence += word + ' '
            output_line = vid + ',' + sentence
            if (i + 1) != len(test_id):
                output_line += '\n'
            f.write(output_line)
            
if __name__ == '__main__':
    main()