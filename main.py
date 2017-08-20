import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

from vgg_fcn import *

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def mean_iou(ground_truth, prediction, num_classes):
    iou, iou_op = tf.metrics.mean_iou(ground_truth, prediction, num_classes)
    return iou, iou_op


def train_nn(sess, epochs, batch_size, train_gen, test_gen, network):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param train_gen: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param test_gen: Function to get batches for validation data
    :param network: The complete VGGFCN network with all data included
    """
    saver = tf.train.Saver(max_to_keep=10)
    merge_all = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    iteration = 0

    for i in range(epochs):
        batch_number = 0
        lrn_rate = 0.0005 if i<10 else 0.0001
        for images, labels in train_gen(batch_size):
            loss, _ = sess.run([network.loss, network.trainer],
                               feed_dict={network.image_input: images,
                                          network.correct_labels: labels,
                                          network.keep_prob: .60,
                                          network.learning_rate: lrn_rate})

            if batch_number % 5 == 0:
                summary, loss = sess.run([merge_all, network.loss],
                                            feed_dict={network.image_input: images,
                                                       network.correct_labels: labels,
                                                       network.keep_prob: 1.0,
                                                       network.learning_rate: lrn_rate})
                writer.add_summary(summary, iteration)

            if batch_number % 10 == 0:
                print("Epoch: {} Batch {}. Loss {:.5f}".format(i+1, batch_number, loss))

            if batch_number % 10 == 0:
                saver.save(sess, "checkpoints/kitty-ep{}-b{}.ckpt".format(i+1,batch_number))

            batch_number += 1
            iteration+=1
        # end of epoch.
    saver.save(sess, "checkpoints/kitty-final.ckpt")


# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    net = VGGFCN(num_classes, data_dir)
    net.download()

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        net.session = sess
        net.load()  # load graph from disk
        net.create_decoder()  # creating decoder and setting up loss
        # Create function to get batches
        train_generator = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        test_generator = helper.gen_batch_function(os.path.join(data_dir, 'data_road/testing'), image_shape)
        sess.run(tf.global_variables_initializer())

        epochs = 20
        batch_size = 16

        train_nn(sess, epochs, batch_size, train_generator, test_generator, net)
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, net.logits, net.keep_prob, net.image_input)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
