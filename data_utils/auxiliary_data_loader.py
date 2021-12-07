from tqdm import *
import random
import logging
import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from .tensor_util import TensorUtil
# from .data_loader import DataLoader

class BaseDataLoader():
   
    def __init__(self, batch_size):
        self.batch_size = batch_size
    
    def make_minibatch_iterator(self, buckets):

        bucket_ids = list(buckets.keys())
        random.shuffle(bucket_ids)
        
        for bucket_idx in bucket_ids:

            bucket_data = buckets[bucket_idx]
            random.shuffle(bucket_data)
            
            elements = []
            samples = 0
      
            for i, ele in enumerate(bucket_data):
                elements.append(ele)
                samples += 1
                if samples >= self.batch_size:     
                    yield elements
                    elements = []
                    samples = 0

class AuxiliaryDataLoader():
    LOGGER = logging.getLogger('AuxiliaryDataLoader')

    def __init__(self,  main_batch_size, auxiliary_batch_size):

        self.main_data_loader = BaseDataLoader(batch_size=main_batch_size)
        self.auxiliary_data_loader = BaseDataLoader(batch_size=auxiliary_batch_size)
        self.tensor_util = TensorUtil()
        
    def make_minibatch_iterator(self, main_buckets, auxiliary_buckets):

        main_batch_iterator = self.main_data_loader.make_minibatch_iterator(main_buckets)
        auxiliary_batch_iterator = self.auxiliary_data_loader.make_minibatch_iterator(auxiliary_buckets)

        for main_batch, aux_batch in zip(main_batch_iterator, auxiliary_batch_iterator):
            batch = main_batch + aux_batch
            batch_obj = self.tensor_util.trees_to_batch_tensors(batch)      
            yield batch_obj


