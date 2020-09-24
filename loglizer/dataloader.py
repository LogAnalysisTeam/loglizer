"""
The interface to load log datasets. The datasets currently supported: HDFS, BGL. 
This file is based on loglizer's dataloader.py.
"""

import os
import pandas as pd
from os.path import join as pjoin
import numpy as np
import re
from datetime import datetime
from statistics import mean, median
from sklearn.utils import shuffle
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)

def _split_data(x_data, y_data=None, id_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        id_pos = id_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        id_neg = id_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        id_train = np.hstack([id_pos[0:train_pos], id_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
        id_test = np.hstack([id_pos[train_pos:], id_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    
    if id_data is not None:
        return (x_train, y_train), (x_test, y_test), (id_train, id_test)
    
    return (x_train, y_train), (x_test, y_test)


def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0, save_path='./'):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    logger.info('====== Input data summary ======')
    logger.info('log_file: {}'.format(log_file))
    logger.info('label_file: {}'.format(label_file))

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        logger.info("Loading {}".format(log_file))
        struct_log = pd.read_csv(log_file, engine='c',
                na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        block_dict = {}
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                    block_dict[blk_Id] = {'blk':blk_Id, 'logs':[], 'timestamps':[]}
                data_dict[blk_Id].append(row['EventId'])
                block_dict[blk_Id]['logs'].append(f"{row['Level']} {row['Component']}: {row['Content']}")
                block_dict[blk_Id]['timestamps'].append(datetime.strptime(f"{row['Date']:06d} {row['Time']:06d}", '%d%m%y %H%M%S').timestamp())

        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)
            for blk_Id in block_dict.keys():
                block_dict[blk_Id]['label'] = label_dict[blk_Id] == 'Anomaly'

            # Split train and test data
            (x_train, y_train), (x_test, y_test), (blk_train, blk_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, data_df['BlockId'], train_ratio, split_type)

            save_for_fulltext_detection(block_dict, blk_train, blk_test, save_path)

            logger.info("y_train sum: {}, y_test sum: {}".format(y_train.sum(), y_test.sum()))

        if save_csv:
            data_df.to_csv(pjoin(save_path, 'data_instances.csv'), index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            logger.info(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            logger.info(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                logger.info('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            logger.info('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    logger.info('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    logger.info('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    logger.info('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)


def slice_hdfs(x, y, window_size):
    results_data = []
    logger.info("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    logger.info("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]


def save_for_fulltext_detection(block_dict, blk_train, blk_test, save_path):
    logger.info('Saving datasets for full text detection')
    import torch
    validation_idx = int(len(blk_train)*0.8)
    np.random.shuffle(blk_train)
    baf = pjoin(save_path, 'benchmark_anomal.pt')
    bf = pjoin(save_path, 'benchmark.pt')
    torch.save({
        'train': [block_dict[blk] for blk in blk_train[0:validation_idx]],
        'validation': [block_dict[blk] for blk in blk_train[validation_idx:]],
        'test': [block_dict[blk] for blk in blk_test]
    }, baf)
    torch.save({
        'train': [block_dict[blk] for blk in blk_train[0:validation_idx] if not block_dict[blk]['label']],
        'validation': [block_dict[blk] for blk in blk_train[validation_idx:] if not block_dict[blk]['label']],
        'test': [block_dict[blk] for blk in blk_test]
    }, bf)


def load_BGL(log_file, save_path='./', window='sliding', time_interval=60, stepping_size=60, 
             train_ratio=0.8):
    """  TODO

    """
    prefile = pjoin(save_path, "train_test_preprocessed.npz")
    if os.path.isfile(prefile):
        logger.info("preprocessed file found, loading {}".format(prefile))
        f = np.load(prefile, allow_pickle=True)
        return (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])
    
    logger.info("loading BGL log file: {}".format(log_file))
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    struct_log['Label'] = struct_log['Label'].apply(lambda x: 0 if x == '-' else 1)
    label_time_pairs = np.array(list(zip(struct_log['Label'],struct_log['Timestamp'])))
    
    logger.info('anomalies: {}'.format(sum(struct_log["Label"])))
    params = {
        'save_path': save_path,
        'window_size': time_interval,
        'step_size': stepping_size
    }

    logger.info("save path for preprocessed files: {}".format(save_path))
    logger.info("preprocessing...")
    event_count_matrix, labels = bgl_preprocess_data(params, label_time_pairs, struct_log)

    logger.info("splitting...")
    (x_train, y_train), (x_test, y_test), (id_train, id_test) = _split_data(event_count_matrix, labels, np.arange(labels.shape[0]), train_ratio)
    # x_train is ndarray of Python lists of event ids 
    logger.info("x_train: {}".format(x_train.shape))
    logger.info("y_train: {}".format(y_train.shape))
    logger.info("x_test: {}".format(x_test.shape))
    logger.info("y_test: {}".format(y_test.shape))
    logger.info("saving to: {}".format(prefile))
    np.savez_compressed(prefile, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    logger.info('Transforming data for full text prediction')
    block_dict = {}
    with open(params['sliding_cache_file']) as f:
        for i, line in enumerate(f):
            block = {'blk':i, 'logs':[], 'timestamps':[], 'labels':[]}
            s,e = line.split(',')
            anomal = False
            for log_id in range(int(s), int(e)):
                anomal |= struct_log['Label'][log_id] == 1
                block['labels'].append(struct_log['Label'][log_id])
                block['timestamps'].append(struct_log['Timestamp'][log_id])
                block['logs'].append(' '.join(struct_log.loc[log_id][['Id2','User','Group','Level','Content']]))
            block['label'] = anomal
            block_dict[i] = block

    save_for_fulltext_detection(block_dict, id_train, id_test, save_path)

    return (x_train, y_train), (x_test, y_test)



def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event sequences and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = pjoin(para['save_path'], 'sliding_{}_h{}h.csv'.format(para['window_size'], para['step_size']))
    para['sliding_cache_file'] = sliding_file_path

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            if end_index-start_index > 2:
                start_end_pair = tuple((start_index, end_index))
                start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        logger.info('there are %d instances (sliding windows) in this dataset\n' % inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        logger.info('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        logger.info('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = event_mapping_data['EventId']
    event_num = len(list(set(event_mapping_data)))
    logger.info('There are %d log events' % event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    sequences = [] #np.empty((inst_number), dtype=object)
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        sequence = []
        for k in expanded_indexes_list[j]:
            sequence.append(event_mapping_data[k])
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
        sequences.append(sequence)
    assert inst_number == len(labels)
    logger.info("Among %d instances, %d are anomalies"%(len(sequences),sum(labels)))
    lenghts = list(map(len,sequences))
    logger.info("Instances len mean=%d median=%d"%(mean(lenghts),median(lenghts)))
    return np.array(sequences), np.array(labels)


