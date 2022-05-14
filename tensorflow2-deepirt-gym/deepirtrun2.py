import logging
import argparse
import numpy as np
from load_data import DataLoader
import tensorflow as tf
import sklearn
from tensorflow.python.keras import metrics
from utils import getLogger
from utils import ProgressBar
from deepirtmodel import deepirt
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
def np_to_dataset(q_data, qa_data, label, shuffle=True, batch_size=32):
    """numpyをdatasetに変換する関数
    Args:
        student(ndarray): 生徒のone-hotデータ
        item(ndarray): 問題のone-hotデータ
        answer(ndarray): 問題に対する反応データ
    Returns:
        object: tensorflowのdatasetクラス(shuffleしたりbatchで取り出すのが楽になる)
    """
    ds = tf.data.Dataset.from_tensor_slices((q_data, qa_data, label))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(q_data))
    ds = ds.batch(batch_size)
    return ds
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


def compute_accuracy(y_true, y_pred):
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
    clipped_filtered_pred = tf.clip_by_value(filtered_pred, epsilon, 1. - epsilon)
    filtered_logits = tf.compat.v1.log(clipped_filtered_pred / (1 - clipped_filtered_pred))
    logit_list = []
    label_list = []
    for i in range(len(filtered_z_values)):
        if filtered_z_values > 0.5:
            logit_list.append(1.0)
        else:
            logit_list.append(0.0)

    for i in range(len(filtered_label)):
        if filtered_label > 0.5:
            label_list.append(1.0)
        else:
            label_list.append(0.0)
    print(logit_list, label_list)
    return sklearn.metrics.accuracy_score(label_list, logit_list)

def sigmoidcrossentropy(y_true, y_pred):
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
    clipped_filtered_pred = tf.clip_by_value(filtered_pred, epsilon, 1. - epsilon)
    filtered_logits = tf.compat.v1.log(clipped_filtered_pred / (1 - clipped_filtered_pred))

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=filtered_logits, labels=filtered_label))



def compute_loss(y_true, y_pred):
    loss = np.array(y_true) * np.log(np.maximum(1e-10, np.array(y_pred))) + \
           (1.0 - np.array(y_pred)) * np.log(np.maximum(1e-10, 1.0 - np.array(y_pred)))
    return np.average(loss) * (-1.0)





def auroc(y_true, y_pred):
    # convert into 1D
    label_1d = tf.reshape(y_true, [-1])
    pred_z_values_1d = tf.reshape(y_pred, [-1])

    # find the label index that is not masking
    index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

    # masking
    filtered_label = tf.gather(label_1d, index)
    filtered_z_values = tf.gather(pred_z_values_1d, index)

    return tf.compat.v1.py_func(roc_auc_score, (filtered_label, filtered_z_values), tf.double)

def compute_auc(y_true, y_pred):
    # convert into 1D
    # label_1d = tf.reshape(y_true, [-1])
    # pred_z_values_1d = tf.reshape(y_pred, [-1])

    #     # find the label index that is not masking
    # index = tf.where(tf.not_equal(label_1d, tf.constant(-1., dtype=tf.float32)))

    #     # masking
    # filtered_label = tf.gather(label_1d, index)
    # filtered_z_values = tf.gather(pred_z_values_1d, index)

    # pred = tf.math.sigmoid(pred_z_values_1d)
    # filtered_pred = tf.math.sigmoid(filtered_z_values)

    #     # convert the prediction probability to logit, i.e., log(p/(1-p))
    # epsilon = 1e-6
    # clipped_filtered_pred = tf.clip_by_value(filtered_pred, epsilon, 1.-epsilon)
    # filtered_logits = tf.compat.v1.log(clipped_filtered_pred/(1-clipped_filtered_pred))
    pred = y_pred.astype(np.double)
    label = y_true.astype(np.double)
    label_batch = (label - 0) // args.n_questions  # convert to {-1, 0, 1}
    label_batch = label_batch.astype(np.double)
    label_flat = label_batch.reshape((-1))
    pred_flat = pred.reshape((-1))

    return tf.compat.v1.py_func(roc_auc_score, (label_flat, pred_flat), tf.double)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def auc(y_true, y_pred):
    return K.sum(s, axis=0)

def binaryEntropy(label, pred):
    loss = label * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - label) * np.log(np.maximum(1e-10, 1.0 - pred))
    return np.average(loss) * (-1.0)

@tf.function
def train_step(model, train_ds):
  for q, qa, la in train_ds:
    with tf.GradientTape() as tape:
      predictions = model(q, qa,  training=True)
      compute_loss1 = compute_loss(la, predictions)
      loss = sigmoidcrossentropy(la, predictions)
      auc = auroc(la, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    logs = 'Epoch={},Loss:{},auc:{} training'
    print(tf.strings.format(logs, (epoch, compute_loss1, auc)))

@tf.function
def valid_step(model, valid_ds):
  for q, qa, la in valid_ds:
    predictions_valid = model(q, qa)
    compute_loss1 = compute_loss(la, predictions_valid)
    loss = sigmoidcrossentropy(la, predictions_valid)
    auc = auroc(la, predictions_valid)
    logs = 'Epoch={},Loss:{},auc:{} valid'
    print(tf.strings.format(logs, (epoch, compute_loss1, auc)))


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
        result_csv_path = os.path.join(args.result_log_dir, 'fold-{}-result'.format(i) + '.csv')

        data_loader = DataLoader(args.n_questions, args.seq_len, ',')
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model = deepirt(value_memory_state_dim=args.value_memory_state_dim,
                        key_memory_state_dim=args.key_memory_state_dim,
                        memory_size=args.memory_size, n_questions=args.n_questions, seq_len=args.seq_len,
                        summary_vector_output_dim=args.summary_vector_output_dim, reuse_flag=True)
        if args.train:
            train_data_path = os.path.join(args.data_dir, args.data_name + '_train{}.csv'.format(i))
            valid_data_path = os.path.join(args.data_dir, args.data_name + '_valid{}.csv'.format(i))
            logger.info("Reading {} and {}".format(train_data_path, valid_data_path))

            train_q_data, train_qa_data = data_loader.load_data(train_data_path)
            valid_q_data, valid_qa_data = data_loader.load_data(valid_data_path)


            training_step = train_q_data.shape[0] // args.batch_size
            train_q_data = train_q_data[: training_step * args.batch_size, :]
            train_qa_data = train_qa_data[: training_step * args.batch_size, :]
            train_label = train_qa_data[:, :]
            train_label = train_label.astype(np.int)
            train_label = (train_label - 1) // args.n_questions
            train_label = train_label.astype(np.float)
            # print(train_q_data.tolist())
            # print(train_qa_data.tolist())
            # print(train_label.tolist())
            train_ds = np_to_dataset(train_q_data , train_qa_data , train_label)

            valid_step = valid_q_data.shape[0] // args.batch_size
            valid_q_data = valid_q_data[: valid_step * args.batch_size, :]
            valid_qa_data = valid_qa_data[: valid_step * args.batch_size, :]
            valid_label = valid_qa_data[:, :]
            valid_label = valid_label.astype(np.int)
            valid_label = (valid_label - 1) // args.n_questions
            valid_label = valid_label.astype(np.float)
            valid_ds = np_to_dataset(valid_q_data, valid_qa_data, valid_label)
            for epoch in range(args.n_epochs):
                train_step(model, train_ds)
                valid_step(model, valid_ds)


if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    cross_validation()

