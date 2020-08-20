import json
import numpy as np
import gzip
import os
from multiprocessing import Process
import mmap

from torch.utils.data import DataLoader, Dataset, TensorDataset, IterableDataset
import torch.distributed as dist
import re


def get_checkpoint_no(checkpoint_path):
    return int(re.findall(r'\d+', checkpoint_path)[-1])


def get_latest_ann_data(ann_data_path):
    if not os.path.exists(ann_data_path):
        return -1, None, None
    files = list(next(os.walk(ann_data_path))[2])
    num_start_pos = len("ann_ndcg_")
    data_no_list = [int(s[num_start_pos:]) for s in files if s[:num_start_pos] == "ann_ndcg_"]
    if len(data_no_list) > 0:
        data_no = max(data_no_list)
        with open(os.path.join(ann_data_path, "ann_ndcg_" + str(data_no)), 'r') as f:
            ndcg_json = json.load(f)
        return data_no, os.path.join(ann_data_path, "ann_training_data_" + str(data_no)), ndcg_json
    return -1, None, None


def numbered_byte_file_generator(base_path, file_no, record_size):
    for i in range(file_no):
        with open('{}_split{}'.format(base_path, i), 'rb') as f:
            while True:
                b = f.read(record_size)
                if not b:
                    # eof
                    break
                yield b


class EmbeddingCache:
    def __init__(self, base_path, seed=-1):
        self.base_path = base_path
        with open(base_path + '_meta', 'r') as f:
            meta = json.load(f)
            self.dtype = np.dtype(meta['type'])
            self.total_number = meta['total_number']
            self.record_size = int(meta['embedding_size']) * self.dtype.itemsize + 4
        mapping_path = base_path + '_idmap.npz'
        # offset2id: array, ix->qid/pid
        self.offset2id = np.load(mapping_path)['idx']
        self.id2offset = {v:i for i, v in enumerate(self.offset2id)}
        assert self.total_number == len(self.offset2id), "Metadata and offset size mismatch! {0} {1}".format(str(self.total_number), str(len(self.offset2id)))
        self.change_seed(seed)
        self.f = None
    
    def change_seed(self, seed):
        if seed>=0:
            self.ix_array = np.random.RandomState(seed).permutation(self.total_number)
        else:
            self.ix_array = np.arange(self.total_number)
        self.seed = seed

    def open(self):
        # f.seek is constant time but random reads are slow due to buffering. disabling buffering 
        # reduces shuffled iteration of 40M elements from 1+ hrs to 4 min (in order iteration takes ~1 min)
        if self.seed>=0:
            self.f = open(self.base_path, 'rb', buffering=0)
        else:
            self.f = open(self.base_path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)

    def close(self):
        self.mm.close()
        self.f.close()

    def read_single_record(self):
        record_bytes = self.mm.read(self.record_size)
        passage_len = int.from_bytes(record_bytes[:4], 'big')
        passage = np.frombuffer(record_bytes[4:], dtype=self.dtype)
        return passage_len, passage

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, key):
        # we key by actual id instead of offset
        if key not in self.id2offset:
            raise IndexError("Key {} is not in mapping".format(key))
        offset = self.id2offset[key]
        self.mm.seek(offset*self.record_size)
        passage_len, passage = self.read_single_record()
        return key, passage_len, passage

    def __iter__(self):
        for ix in self.ix_array:
            real_id = self.offset2id[ix]
            yield self.__getitem__(real_id)

    def __len__(self):
        return self.total_number


class StreamingDataset(IterableDataset):
    def __init__(self, elements, fn):
        super().__init__()
        self.elements = elements
        self.fn = fn
        self.num_replicas=-1 
    
    def __iter__(self):
        if dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
            print("Rank:", self.rank, "world:", self.num_replicas)
        else:
            print("Not running in distributed mode")
        for i, element in enumerate(self.elements):
            if self.num_replicas != -1 and i % self.num_replicas != self.rank:
                continue
            records = self.fn(element, i)
            for rec in records:
                yield rec
