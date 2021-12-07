import sys
import os
infercode_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(infercode_dir)
import logging
from data_utils.ast_util import ASTUtil
from data_utils.token_vocab_extractor import TokenVocabExtractor
from data_utils.dataset_processor import DatasetProcessor
from data_utils.threaded_iterator import ThreadedIterator
from data_utils.data_loader import DataLoader
from network.tbcnn import TreebasedCNN
from data_utils.vocabulary import Vocabulary
from data_utils.language_util import LanguageUtil
import tensorflow.compat.v1 as tf
from .base_client import BaseClient
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
tf.disable_v2_behavior()

class SoftmaxTrainer(BaseClient):

    LOGGER = logging.getLogger('SoftmaxTrainer')

    def __init__(self, language, train_input_data_path, train_output_processed_data_path, test_input_data_path, test_output_processed_data_path):
        self.language = language
        self.train_input_data_path = train_input_data_path
        self.train_output_processed_data_path = train_output_processed_data_path

        self.test_input_data_path = test_input_data_path
        self.test_output_processed_data_path = test_output_processed_data_path


    def init_from_config(self, config=None):
        # Load default config if do not provide an external one
        self.load_configs(config)
        self.init_params()
        self.init_resources()
        self.init_utils()
        self.init_model_checkpoint()

        # ------------Set up the neural network------------
       
        self.training_data_processor = DatasetProcessor(input_data_path=self.train_input_data_path, 
                                                       output_tensors_path=self.train_output_processed_data_path, 
                                                       node_type_vocab_model_prefix=self.node_type_vocab_model_prefix, 
                                                       node_token_vocab_model_prefix=self.node_token_vocab_model_prefix, 
                                                       language=self.language)
        self.training_buckets = self.training_data_processor.process_or_load_data()

        self.testing_data_processor = DatasetProcessor(input_data_path=self.test_input_data_path, 
                                                       output_tensors_path=self.test_output_processed_data_path, 
                                                       node_type_vocab_model_prefix=self.node_type_vocab_model_prefix, 
                                                       node_token_vocab_model_prefix=self.node_token_vocab_model_prefix, 
                                                       language=self.language)

        self.testing_buckets = self.testing_data_processor.process_or_load_data()

        # self.ast_util, self.training_buckets = self.process_or_load_data()        
        self.data_loader = DataLoader(self.batch_size)
            
        # ------------Set up the neural network------------
        self.tbcnn = TreebasedCNN(num_types=self.node_type_vocab.get_vocabulary_size(), 
                                  num_tokens=self.node_token_vocab.get_vocabulary_size(), 
                                  num_labels=self.num_labels,
                                  loss_type=self.loss_type,
                                  num_conv=self.num_conv, 
                                  node_type_dim=self.node_type_dim, 
                                  node_token_dim=self.node_token_dim,
                                  conv_output_dim=self.conv_output_dim, 
                                  include_token=self.include_token, 
                                  batch_size=self.batch_size, 
                                  learning_rate=self.learning_rate)

        self.saver = tf.train.Saver(save_relative_paths=True, max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)

        self.checkfile = os.path.join(self.model_checkpoint, 'cnn_tree.ckpt')
        ckpt = tf.train.get_checkpoint_state(self.model_checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            self.LOGGER.info("Load model successfully : " + str(self.checkfile))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.LOGGER.error("Could not find the model : " + str(self.checkfile))
            self.LOGGER.error("Train the model from scratch")
        
        # -------------------------------------------------

    def train(self):
        best_f1 = 0.0
        for epoch in range(1,  self.epochs + 1):
            train_batch_iterator = ThreadedIterator(self.data_loader.make_minibatch_iterator(self.training_buckets), max_queue_size=5)
            for train_step, train_batch_data in enumerate(train_batch_iterator):

                print(train_batch_data["batch_labels"])
                _, err = self.sess.run(
                    [self.tbcnn.training_point,
                    self.tbcnn.loss_avg],
                    feed_dict={
                        self.tbcnn.placeholders["node_type"]: train_batch_data["batch_node_type_id"],
                        self.tbcnn.placeholders["node_tokens"]:  train_batch_data["batch_node_tokens_id"],
                        self.tbcnn.placeholders["children_index"]:  train_batch_data["batch_children_index"],
                        self.tbcnn.placeholders["children_node_type"]: train_batch_data["batch_children_node_type_id"],
                        self.tbcnn.placeholders["children_node_tokens"]: train_batch_data["batch_children_node_tokens_id"],
                        self.tbcnn.placeholders["labels"]: train_batch_data["batch_labels"],
                        self.tbcnn.placeholders["dropout_rate"]: 0.3
                    }
                )

                self.LOGGER.info(f"Epoch: {epoch} , Step: {train_step} , Loss: {err}, Best F1: {best_f1}")                
                if train_step % self.checkpoint_every == 0:
                    test_batch_iterator = ThreadedIterator(self.data_loader.make_minibatch_iterator(self.testing_buckets), max_queue_size=5)

                    correct_labels = []
                    predictions = []
                    for test_step, test_batch_data in enumerate(test_batch_iterator):

                        predicted_labels = self.sess.run(
                                [self.tbcnn.predicted_labels],
                                feed_dict={
                                    self.tbcnn.placeholders["node_type"]: test_batch_data["batch_node_type_id"],
                                    self.tbcnn.placeholders["node_tokens"]:  test_batch_data["batch_node_tokens_id"],
                                    self.tbcnn.placeholders["children_index"]:  test_batch_data["batch_children_index"],
                                    self.tbcnn.placeholders["children_node_type"]: test_batch_data["batch_children_node_type_id"],
                                    self.tbcnn.placeholders["children_node_tokens"]: test_batch_data["batch_children_node_tokens_id"],
                                    self.tbcnn.placeholders["labels"]: test_batch_data["batch_labels"],
                                    self.tbcnn.placeholders["dropout_rate"]: 0.0
                                }
                            )
                
                        correct_labels.extend(test_batch_data["batch_labels"])
                        predictions.extend(predicted_labels[0])

                    f1 = float(f1_score(correct_labels, predictions, average="micro"))
                    self.LOGGER.info(f"F1 score at {epoch} and step {train_step} with value: {f1}")
                    if f1 > best_f1:
                        best_f1 = f1
                        self.saver.save(self.sess, self.checkfile)                  
                        self.LOGGER.info(f"Checkpoint saved, epoch {epoch} and step {train_step} with loss {err}")
