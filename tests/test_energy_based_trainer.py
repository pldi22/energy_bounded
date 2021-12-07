import sys
from pathlib import Path
# To import upper level modules
sys.path.append(str(Path('.').absolute().parent))
from client.energy_trainer import EnergyTrainer
import configparser
import logging
import pathlib
print(pathlib.Path().resolve())
logging.basicConfig(level=logging.INFO)

# config = configparser.ConfigParser()
# config.read("../configs/OJ_raw_small.ini")
train_input_data_path = "/home/nghibui/codes/datasets/OJ104_small_train_test_val/train"
train_output_processed_data_path = "/home/nghibui/codes/datasets/OJ104_small_train_test_val/train.pkl"

train_aux_input_data_path = "/home/nghibui/codes/datasets/ood/train"
train_aux_output_processed_data_path = "/home/nghibui/codes/datasets/ood/train.pkl"

test_input_data_path = "/home/nghibui/codes/datasets/OJ104_small_train_test_val/test"
test_output_processed_data_path = "/home/nghibui/codes/datasets/OJ104_small_train_test_val/test.pkl"

test_aux_input_data_path = "/home/nghibui/codes/datasets/ood/test"
test_aux_output_processed_data_path = "/home/nghibui/codes/datasets/ood/test.pkl"

infercode_trainer = EnergyTrainer(language="c", 
                                train_input_data_path=train_input_data_path,
                                train_output_processed_data_path=train_output_processed_data_path,
                                train_aux_input_data_path=train_aux_input_data_path,
                                train_aux_output_processed_data_path=train_aux_output_processed_data_path,
                                test_input_data_path=test_input_data_path,
                                test_output_processed_data_path=test_output_processed_data_path,
                                test_aux_input_data_path=test_aux_input_data_path,
                                test_aux_output_processed_data_path=test_aux_output_processed_data_path)
infercode_trainer.init_from_config()
infercode_trainer.train()