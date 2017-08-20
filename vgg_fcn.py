"""A Wrapper class for the Full Convolutional Network implementation of VGG Net"""
# Author: Kostas Oreopoulos

import zipfile
import shutil
import os
import tensorflow as tf

from tqdm import tqdm
from urllib.request import urlretrieve


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def make_1x1_convolution(input_layer, n_classes, scope, name, padding='SAME'):
    with tf.variable_scope(scope):
        layer_1x1 = tf.layers.conv2d(inputs=input_layer,
                                     filters=n_classes,
                                     kernel_size=1,
                                     padding=padding,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                     name=name)
    return layer_1x1


def upsample(layer, n_classes, kernel_size, strides, scope, name, padding='SAME'):
    with tf.variable_scope(scope):
        upsampled = tf.layers.conv2d_transpose(inputs=layer,
                                               filters=n_classes,
                                               kernel_size=kernel_size,
                                               strides=strides,
                                               padding=padding,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                               name=name)
    return upsampled


class VGGFCN:
    def __init__(self, classes, data_dir):
        self.num_classes = classes
        self.data_dir = data_dir
        self.session = None
        self.image_input = None
        self.keep_prob = None
        self.learning_rate = None
        self.softmax = None
        self.layer3 = None
        self.layer4 = None
        self.layer7 = None
        self.logits = None
        self.labels = None
        self.correct_labels = None
        self.loss = None
        self.trainer = None

    def download(self):
        """
        Download and extract pretrained vgg model if it doesn't exist
        """
        vgg_filename = 'vgg.zip'
        vgg_path = os.path.join(self.data_dir, 'vgg')
        vgg_files = [
            os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
            os.path.join(vgg_path, 'variables/variables.index'),
            os.path.join(vgg_path, 'saved_model.pb')]

        missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
        if missing_vgg_files:
            # Clean vgg dir
            if os.path.exists(vgg_path):
                shutil.rmtree(vgg_path)
            os.makedirs(vgg_path)

            # Download vgg
            print('Downloading pre-trained vgg model...')
            with DLProgress(unit='B', unit_scale=True, miniters=1) as p_bar:
                urlretrieve(
                    'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                    os.path.join(vgg_path, vgg_filename),
                    p_bar.hook)

            # Extract vgg
            print('Extracting model...')
            zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
            zip_ref.extractall(self.data_dir)
            zip_ref.close()

            # Remove zip file to save space
            os.remove(os.path.join(vgg_path, vgg_filename))

    def load(self):
        if self.data_dir is None or self.session is None:
            raise Exception('Data Directory for VGG was not set')
        sess = self.session
        vgg_tag = 'vgg16'
        vgg_path = self.data_dir + '/vgg'
        # read graph and relevant layers
        graph = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')
        self.layer3 = sess.graph.get_tensor_by_name('layer3_out:0')
        self.layer4 = sess.graph.get_tensor_by_name('layer4_out:0')
        self.layer7 = sess.graph.get_tensor_by_name('layer7_out:0')

    def create_decoder(self):
        layer7_1x1 = make_1x1_convolution(self.layer7, self.num_classes, 'decoder', 'L7_1x1')
        layer4_1x1 = make_1x1_convolution(self.layer4, self.num_classes, 'decoder', 'L4_1x1')
        layer3_1x1 = make_1x1_convolution(self.layer3, self.num_classes, 'decoder', 'L3_1x1')

        decoder_layer7 = upsample(layer7_1x1, self.num_classes, kernel_size=(4, 4), strides=(2, 2), scope='decoder',
                                  name='Up7')

        skip_4to7 = tf.add(layer4_1x1, decoder_layer7)
        decoder_4and7 = upsample(skip_4to7, self.num_classes, kernel_size=(4, 4), strides=(2, 2), scope='decoder',
                                 name='skip_4to7')

        skip_3_to_decoder_4and7 = tf.add(layer3_1x1, decoder_4and7)
        decoder_3and4and7 = upsample(skip_3_to_decoder_4and7, self.num_classes, kernel_size=(16, 16), strides=(8, 8),
                                     scope='decoder', name='3and4and7')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.correct_labels = tf.placeholder(tf.float32, shape=(None, None, None, self.num_classes), name='labels')
        self.labels = tf.reshape(self.correct_labels, [-1, self.num_classes])

        with tf.variable_scope('result'):
            self.logits = tf.reshape(decoder_3and4and7, [-1, self.num_classes], name='logits')
            self.softmax = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels, name='softmax')
        with tf.variable_scope('train'):
            self.loss = tf.reduce_mean(self.softmax, name='loss')
            self.trainer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # tensorboard
        tf.summary.histogram("logits", self.logits)
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("softmax", self.softmax)
