import glob
import torch
import random
import logging

from random import choices
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from sentence_transformers.readers import InputExample

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

class PairData(object):
    def __init__(self, fpath):
        self.fpath = fpath
        
    def get_example(self, shuffle, duplicates=1, num_data=-1, gpu=None, _type="bi_enc"):
        """
        ratio : example내 pos의 비율
        """
        fnames = random.sample(glob.glob("{}/*".format(self.fpath)), 1)[0]
        with open(fnames) as f:
            lines = f.readlines()
        data = []
        for line in lines:
            label, q1, q2, inter_cnt, npmi, data_type = line.rstrip('\n').split('\t')
            if _type == "bi_enc":
                example = InputExample(texts=[q1, q2], label=int(label))
            # elif _type == "cross_enc":
            #     example = InputExample(texts=[q1, q2], label=float(npmi))
            if label == '1':
                data.extend([example] * duplicates)
            else:
                data.append(example)
        if shuffle:
            random.shuffle(data)
        logger.debug("{} loaded on GPU {}.".format(fnames, gpu))
        if num_data == -1:
            return data
        else:
            return data[:num_data]
    
    def get_data_iter(self, batch_size, is_train, duplicates, gpu):
        while True:
            examples = self.get_example(is_train, duplicates=duplicates, gpu=gpu)
            if is_train:
                sampler = DistributedSampler(examples)
                yield DataLoader(examples, batch_size=batch_size, sampler=sampler)
            else:
                yield DataLoader(examples, batch_size=batch_size)