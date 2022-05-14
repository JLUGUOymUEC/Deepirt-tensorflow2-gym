import logging
import argparse
import numpy as np
from load_data_1 import DataLoader
import tensorflow as tf
import sklearn
from tensorflow.python.keras import metrics
from utils import getLogger
from utils import ProgressBar
from deepirtmodel import deepirt_dropout
from configs import ModelConfigFactory
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os
import tqdm
import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



# set logger
logger = getLogger('Deep-IRT-model')

# argument parser
parser = argparse.ArgumentParser()
# dataset can be assist2009, assist2015, statics2011, synthetic, fsai
parser.add_argument('--dataset', default='assist2009',
                    help="'assist2009', 'assist2015', 'statics2011', 'synthetic', 'fsai'")

parser.add_argument('--save', type=bool, default=False)
parser.add_argument('--cpu', type=bool, default=False)
parser.add_argument('--n_epochs', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--train', type=bool, default=None)
parser.add_argument('--show', type=bool, default=None)
parser.add_argument('--learning_rate', type=float, default=None)
parser.add_argument('--max_grad_norm', type=float, default=None)
parser.add_argument('--use_ogive_model', type=bool, default=False)

# parameter for the dataset
parser.add_argument('--seq_len', type=int, default=None)
parser.add_argument('--n_questions', type=int, default=None)
parser.add_argument('--n_skills', type=int, default=None)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--data_name', type=str, default=None)

# parameter for the DKVMN model
parser.add_argument('--memory_size', type=int, default=None)
parser.add_argument('--key_memory_state_dim', type=int, default=None)
parser.add_argument('--value_memory_state_dim', type=int, default=None)
parser.add_argument('--summary_vector_output_dim', type=int, default=None)

_args = parser.parse_args()
args = ModelConfigFactory.create_model_config(_args)
logger.info("Model Config: {}".format(args))

def compute_loss(y_true,y_pred):
  label_1d = tf.reshape(y_true, [-1])
  pred_z_values_1d = tf.reshape(y_pred, [-1])


  # find the label index that is not masking
  index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

  # masking
  filtered_label = tf.gather(label_1d, index)
  filtered_z_values = tf.gather(pred_z_values_1d, index)
  loss = filtered_label * tf.compat.v1.log(np.maximum(1e-10,filtered_z_values)) + \
        (1.0 - filtered_label) * tf.compat.v1.log( np.maximum(1e-10, 1.0-filtered_z_values) )
  return np.average(loss)*(-1.0)


def compute_accuracy(y_true,y_pred):
    label_flat = tf.reshape(y_true, [-1])
    pred_flat = tf.reshape(y_pred, [-1])
    index_flat = tf.where(tf.not_equal(label_flat, tf.constant(-1., dtype=tf.float32)))

    label_flat_1 = label_flat[index_flat]
    pred_flat_1 = pred_flat[index_flat]
    return tf.keras.metrics.binary_accuracy(label_flat_1, pred_flat_1, threshold = 0.5)

# def compute_accuracy(y_true,y_pred):
#     label_flat = tf.reshape(y_true, [-1])
#     pred_flat = tf.reshape(y_pred, [-1])
#     index_flat = tf.where(tf.not_equal(label_flat, tf.constant(-1., dtype=tf.float32)))

#     label_flat_1 = label_flat[index_flat]
#     pred_flat_1 = pred_flat[index_flat]
#     return tf.keras.metrics.Accuracy(label_flat_1, tf.where(pred_flat_1 >0.5, x=1.0, y=0.0))

def sigmoidcrossentropy(y_true,y_pred):
      # convert into 1D
      label_1d = tf.reshape(y_true, [-1])
      pred_z_values_1d = tf.reshape(y_pred, [-1])


        # find the label index that is not masking
      index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

        # masking
      filtered_label = tf.gather(label_1d, index)
      filtered_z_values = tf.gather(pred_z_values_1d, index)

      pred = tf.math.sigmoid(pred_z_values_1d)
      filtered_pred = tf.math.sigmoid(filtered_z_values)

        # convert the prediction probability to logit, i.e., log(p/(1-p))
      epsilon = 1e-6
      clipped_filtered_pred = tf.clip_by_value(filtered_pred, epsilon, 1.-epsilon)
      filtered_logits = tf.compat.v1.log(clipped_filtered_pred/(1-clipped_filtered_pred))

      return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_label))
def auroc(y_true, y_pred):
    label_flat = tf.reshape(y_true, [-1])
    pred_flat = tf.reshape(y_pred, [-1])
    index_flat = tf.where(tf.not_equal(label_flat, tf.constant(-1., dtype=tf.float32)))

    label_flat_1 = label_flat[index_flat]
    pred_flat_1 = pred_flat[index_flat]
    # # convert into 1D
    # label_1d = tf.reshape(y_true, [-1])
    # pred_z_values_1d = tf.reshape(y_pred, [-1])


    #     # find the label index that is not masking
    # index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

    #     # masking
    # filtered_label = tf.gather(label_1d, index)
    # filtered_z_values = tf.gather(pred_z_values_1d, index)

    return  tf.compat.v1.py_func(roc_auc_score, (label_flat_1, pred_flat_1), tf.double)

def compute_auc(y_true,y_pred):
    pred = y_pred.astype(np.double)
    label = y_true.astype(np.double)
    label_batch = (label - 0) // args.n_skills  # convert to {-1, 0, 1}
    label_batch = label_batch.astype(np.double)
    label_flat = label_batch.reshape((-1))
    pred_flat = pred.reshape((-1))

    return tf.compat.v1.py_func(roc_auc_score, (label_flat, pred_flat), tf.double)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def auc(y_true, y_pred):

    return K.sum(s, axis=0)

# def binaryEntropy(label, pred):
#     loss = label * np.log(np.maximum(1e-10,pred)) + \
#            (1.0 - label) * np.log( np.maximum(1e-10, 1.0-pred) )
#     return np.average(loss)*(-1.0)

def binaryEntropy(label, pred):

     return tf.keras.losses.binary_crossentropy(label, pred)

def cross_validation():
    tf.random.set_seed(1234)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    # if args.cpu:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    aucs, accs, losses = list(), list(), list()
    for i in range(5):
        tf.compat.v1.disable_v2_behavior()
        logger.info("Cross Validation {}".format(i + 1))
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i+1) + '.csv')

        data_loader = DataLoader(args.n_questions,args.n_skills, args.seq_len, ',')
        optimizer = tf.keras.optimizers.Adam(learning_rate= args.learning_rate)
        model = deepirt_dropout(value_memory_state_dim = args.value_memory_state_dim, key_memory_state_dim = args.key_memory_state_dim,
                        memory_size = args.memory_size, n_questions = args.n_questions, seq_len = args.seq_len,
                        summary_vector_output_dim = args.summary_vector_output_dim,n_skills=args.n_skills,batch_size = args.batch_size, reuse_flag=True)
        model.compile(optimizer= optimizer, loss = sigmoidcrossentropy, metrics = [ auroc,compute_accuracy ])
        if args.train:
                train_data_path = os.path.join(args.data_dir, args.data_name + '_train{}.csv'.format(i+1))
                valid_data_path = os.path.join(args.data_dir, args.data_name + '_valid{}.csv'.format(i+1))
                logger.info("Reading {} and {}".format(train_data_path, valid_data_path))
                best_loss = 1e6
                best_acc = 0.0
                best_auc = 0.0
                best_epoch = 0.0
                train_s_data, train_q_data, train_qa_data = data_loader.load_data(train_data_path)
                valid_s_data, valid_q_data, valid_qa_data = data_loader.load_data(valid_data_path)
                train_writer = tf.summary.create_file_writer(args.result_log_dir)
                for epoch in range(args.n_epochs):

                  shuffle_index_train = np.random.permutation(train_q_data.shape[0])
                  train_q_data = train_q_data[shuffle_index_train]
                  train_qa_data = train_qa_data[shuffle_index_train]
                  train_s_data = train_s_data[shuffle_index_train]
                  training_step = train_q_data.shape[0] // args.batch_size
                  train_q_data = train_q_data[: training_step * args.batch_size, :]
                  train_qa_data = train_qa_data[: training_step * args.batch_size, :]
                  train_s_data = train_s_data[: training_step * args.batch_size, :]
                  train_label = train_qa_data[:, :]
                  train_label = train_label.astype(np.int32)
                  train_label = (train_label - 1)//args.n_skills
                  train_label = train_label.astype(np.float64)


                  shuffle_index_valid = np.random.permutation(valid_q_data.shape[0])
                  valid_q_data = valid_q_data
                  valid_qa_data = valid_qa_data
                  valid_s_data = valid_s_data
                  valid_step = valid_q_data.shape[0] // args.batch_size
                  valid_q_data = valid_q_data[: valid_step * args.batch_size, :]
                  valid_qa_data = valid_qa_data[: valid_step * args.batch_size, :]
                  valid_s_data = valid_s_data[: valid_step * args.batch_size, :]
                  valid_label = valid_qa_data[:, :]
                  valid_label = valid_label.astype(np.int32)
                  valid_label = (valid_label - 1)//args.n_skills
                  valid_label = valid_label.astype(np.float64)
                  print('epoche:' + str(epoch+1))
                  model.fit(x = [np.asarray(train_q_data), np.asarray(train_qa_data), np.asarray(train_s_data)], y = np.asarray(train_label),batch_size= args.batch_size ,epochs = 1  )
                  if epoch == 0:
                    model.summary()
                  loss_metric = model.evaluate(x = [np.asarray(valid_q_data), np.asarray(valid_qa_data), np.asarray(valid_s_data)], y = np.asarray(valid_label), batch_size= args.batch_size)
                  msg = "\n[Epoch {}/{}] Validation result:    AUC: {:.2f}%\t Acc: {:.2f}%\t Loss: {:.4f}%\t ".format(
                      epoch+1, args.n_epochs, loss_metric[1]*100, loss_metric[2]*100, loss_metric[0]*10
                  )
                  logger.info(msg)
                  valid_loss = loss_metric[0]*10
                  valid_accuracy = loss_metric[2]*100
                  valid_auc = loss_metric[1]*100
                  if best_auc < valid_auc:
                    best_loss = valid_loss
                    best_acc = valid_accuracy
                    best_auc = valid_auc
                    best_epoch = epoch+1
                msg = "Best result at epoch {}: AUC: {:.2f}\t Accuracy: {:.2f}\t Loss: {:.4f}\t ".format(
                  best_epoch, best_auc, best_acc, best_loss
                )
                logger.info(msg)
                aucs.append(best_auc)
                accs.append(best_acc)
                losses.append(best_loss)
                # auc, acc, loss = train(
                #     model,
                #     train_q_data, train_qa_data,
                #     valid_q_data, valid_qa_data,
                #     result_log_path=result_csv_path,
                #     args=args
                # )
    cross_validation_msg = "Cross Validation Result:\n"
    cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs), np.std(aucs))
    cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs), np.std(accs))
    cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    logger.info(cross_validation_msg)  

    # cross_validation_msg = "Cross Validation Result:\n"
    # cross_validation_msg += "AUC: {:.2f} +/- {:.2f}\n".format(np.average(aucs) * 100, np.std(aucs) * 100)
    # cross_validation_msg += "Accuracy: {:.2f} +/- {:.2f}\n".format(np.average(accs) * 100, np.std(accs) * 100)
    # cross_validation_msg += "Loss: {:.2f} +/- {:.2f}\n".format(np.average(losses), np.std(losses))
    # logger.info(cross_validation_msg)

    # # write result
    # result_msg = datetime.datetime.now().strftime("%Y-%m-%dT%H%M") + ','
    # result_msg += str(args.dataset) + ','
    # result_msg += str(args.memory_size) + ','
    # result_msg += str(args.key_memory_state_dim) + ','
    # result_msg += str(args.value_memory_state_dim) + ','
    # result_msg += str(args.summary_vector_output_dim) + ','
    # result_msg += str(np.average(aucs) * 100) + ','
    # result_msg += str(np.std(aucs) * 100) + ','
    # result_msg += str(np.average(accs) * 100) + ','
    # result_msg += str(np.std(accs) * 100) + ','
    # result_msg += str(np.average(losses)) + ','
    # result_msg += str(np.std(losses)) + '\n'
    # with open('results/all_result.csv', 'a') as f:
    #     f.write(result_msg)


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    cross_validation()

