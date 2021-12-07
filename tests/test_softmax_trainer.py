import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from client.softmax_trainer import SoftmaxTrainer
import configparser
import logging
import pathlib
print(pathlib.Path().resolve())
logging.basicConfig(level=logging.INFO)

# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")
train_input_data_path = "/home/nghibui/codes/datasets/OJ_raw_train_test_val/train"
train_output_processed_data_path = "/home/nghibui/codes/datasets/OJ_raw_train_test_val/train.pkl"

test_input_data_path = "/home/nghibui/codes/datasets/OJ_raw_train_test_val/val"
test_output_processed_data_path = "/home/nghibui/codes/datasets/OJ_raw_train_test_val/val.pkl"

infercode_trainer = SoftmaxTrainer(language="c", 
                                    train_input_data_path=train_input_data_path,
                                    train_output_processed_data_path=train_output_processed_data_path,
                                    test_input_data_path=test_input_data_path,
                                    test_output_processed_data_path=test_output_processed_data_path)
infercode_trainer.init_from_config()
infercode_trainer.train()