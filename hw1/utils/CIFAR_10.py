import pickle
import numpy as np

def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.

    return results

def unpickle_all_file():
    batch_path = ['CIFAR10/data_batch_' + str(i) for i in range(1, 6)]
    batch_path.append('CIFAR10/test_batch')
    file_list = [open(file_name, 'rb') for file_name in batch_path]
    train_data = []
    train_label = []
    # load data
    for f in file_list[0:5]:
        data = pickle.load(f, encoding='bytes')
        train_data.append(data[b'data'].reshape(10000, 32, 32, 3))
        train_label.append(data[b'labels'])
        f.close()
    test_data_dict = pickle.load(file_list[5], encoding='bytes')
    file_list[5].close()
    batch_data_dict = {'data': train_data, 'labels': train_label}
    # process data
    train_data_dict = {}
    train_data_dict['data'] = np.vstack(batch_data_dict['data'])
    train_data_dict['labels'] = [] 
    test_data_dict['data'] = (test_data_dict.pop(b'data')).reshape(10000, 32, 32, 3)
    train_data_dict['labels'] = to_one_hot(np.array(batch_data_dict['labels']).reshape((50000, 1)), 10)
    test_data_dict['labels'] = to_one_hot(np.array(test_data_dict.pop(b'labels')).reshape((10000, 1)), 10)
    
    return train_data_dict, test_data_dict