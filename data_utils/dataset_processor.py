import numpy as np
import os
from tqdm import *
#import pickle
from .vocabulary import Vocabulary
from .ast_util import ASTUtil
from .tensor_util import TensorUtil
from .ast_parser import ASTParser
from .language_util import LanguageUtil
from .token_vocab_extractor import TokenVocabExtractor
from collections import defaultdict
import pickle
import logging

class DatasetProcessor():
    
    LOGGER = logging.getLogger('DatasetProcessor')
    def __init__(self, input_data_path: str, output_tensors_path: str, 
                node_type_vocab_model_prefix: str,
                node_token_vocab_model_prefix: str, 
                language: str, is_auxiliary: bool = False, max_ast_size: int=1000):
        
        self.language = language
        self.is_auxiliary = is_auxiliary
        self.max_ast_size = max_ast_size
        self.input_data_path = input_data_path
        self.output_tensors_path = output_tensors_path
        self.node_type_vocab_model_prefix = node_type_vocab_model_prefix
        self.node_token_vocab_model_prefix = node_token_vocab_model_prefix

        self.ast_parser = ASTParser(language=language)
        self.language_util = LanguageUtil()

        self.token_vocab_extractor = TokenVocabExtractor(node_token_vocab_model_prefix=self.node_token_vocab_model_prefix, 
                                                        model_type="word")
     
        self.init_vocabs()
        self.tensor_util = TensorUtil()

        # AST Util can only be initialized after extracted the token vocab
        self.ast_util = ASTUtil(node_type_vocab_model_path=self.node_type_vocab_model_prefix + ".model", 
                                node_token_vocab_model_path=self.node_token_vocab_model_prefix + ".model")


    def detect_language_of_file(self, file_path: str):
        _, file_extension = os.path.splitext(file_path)
        return self.language_util.get_language_by_file_extension(file_extension)

    # Trees with similar size should be put into the same bucket
    def put_trees_into_buckets(self):

        bucket_sizes = np.array(list(range(20 , 7500 , 20)))
        buckets = defaultdict(list)
        
        for subdir , dirs, files in os.walk(self.input_data_path): 
            for file in tqdm(files):
                file_path = os.path.join(subdir, file)
              
                with open(file_path, "rb") as f:
                    code_snippet = f.read()
                
                ast = self.ast_parser.parse(code_snippet)
                tree_representation, tree_size = self.ast_util.simplify_ast(ast, code_snippet.decode("utf-8"))

                if tree_size <= self.max_ast_size:
                    tree_indexes = self.tensor_util.transform_tree_to_index(tree_representation)
                    tree_indexes["size"] = tree_size

                    if self.is_auxiliary:
                        tree_indexes["label"] = -1
                    else:
                        file_path_splits = file_path.split("/")           
                        label = int(file_path_splits[len(file_path_splits)-2]) - 1
                        tree_indexes["label"] = label 
                
                    chosen_bucket_idx = np.argmax(bucket_sizes > tree_size)
                    buckets[chosen_bucket_idx].append(tree_indexes)

        self.LOGGER.info("Saving processed data into pickle format.....")
        pickle.dump(buckets, open(self.output_tensors_path, "wb" ) )

        return buckets


    def init_vocabs(self):
        self.LOGGER.info("Loading type vocab from path : " + str(self.node_type_vocab_model_prefix + ".model"))
        if not os.path.exists(self.node_token_vocab_model_prefix + ".model"):
            self.LOGGER.info("Generating token vocabulary")
            self.token_vocab = self.token_vocab_extractor.create_vocab_from_dir(self.input_data_path)
        else:
            self.LOGGER.info("Loading token vocab from path : " + str(self.node_token_vocab_model_prefix + ".model"))
            self.token_vocab = Vocabulary(1000000, self.node_token_vocab_model_prefix + ".model")

    def process_or_load_data(self):

        if not os.path.exists(self.output_tensors_path):
            self.LOGGER.info("Processing the dataset : " + self.input_data_path)
            training_buckets = self.put_trees_into_buckets()
        else:
            self.LOGGER.info("Loading the dataset pickle : " + self.input_data_path)
            training_buckets = pickle.load(open(self.output_tensors_path, "rb"))

        return training_buckets