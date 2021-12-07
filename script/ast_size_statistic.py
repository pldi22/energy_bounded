import os
from tqdm import *
import sys
#import pickle
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from data_utils.ast_parser import ASTParser
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, required=True)
parser.add_argument('--language', type=str, default="c")
parser.add_argument('--output_path', type=str, default="ast_size.csv")

def main(opt):
    path = opt.input_path
    language = opt.language
    output_path = opt.output_path

    ast_parser = ASTParser(language=language)
    sizes = []
    for subdir , dirs, files in os.walk(path): 
        for file in tqdm(files):
            file_path = os.path.join(subdir, file)
          
            with open(file_path, "rb") as f:
                code_snippet = f.read()
            
            ast = ast_parser.parse(code_snippet)
            ast_size = ast_parser.get_ast_size(ast)
            sizes.append(ast_size)

    sizes.sort(reverse=True)

    np.savetxt(output_path, sizes, newline=",", fmt="%s")

           


# python3 extract_token_vocab.py --data_path ../../datasets/OJ_raw_small/ --node_token_vocab_model_prefix OJ_raw_token

if __name__ == "__main__":
    opt = parser.parse_args()
    main(opt)
