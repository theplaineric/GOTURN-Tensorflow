# Date: Friday 21 July 2017
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Training the tracker

from ..helper import config
import argparse
import setproctitle
from ..logger.logger import setup_logger
from ..loader.loader_imagenet import loader_imagenet
from ..loader.loader_alov import loader_alov
from ..train.example_generator import example_generator
from ..network.regressor_train import regressor_train
from ..tracker.tracker_trainer import tracker_trainer
import os
import numpy as np
from .EntryGenerator import EntryGenerator

setproctitle.setproctitle('TRAIN_TRACKER_IMAGENET_ALOV')
logger = setup_logger(logfile=None)
logger.info('Caffe path = {}'.format(config.CAFFE_PATH))

ap = argparse.ArgumentParser()
ap.add_argument("-imagenet", "--imagenet", required=True, help="Path to ImageNet folder")
ap.add_argument("-alov", "--alov", required=True, help="Path to Alov folder")
ap.add_argument("-init_caffemodel", "--init_caffemodel", required=True, help="Path to caffe Init model")
ap.add_argument("-train_prototxt", "--train_prototxt", required=True, help="train prototxt")
ap.add_argument("-solver_prototxt", "--solver_prototxt", required=True, help="solver prototxt")
ap.add_argument("-lamda_shift", "--lamda_shift", required=True, help="lamda shift")
ap.add_argument("-lamda_scale", "--lamda_scale", required=True, help="lamda scale ")
ap.add_argument("-min_scale", "--min_scale", required=True, help="min scale")
ap.add_argument("-max_scale", "--max_scale", required=True, help="max scale")
ap.add_argument("-gpu_id", "--gpu_id", required=True, help="gpu id")


RANDOM_SEED = 800
GPU_ONLY = True
kNumBatches = 500000


# TODO: create new class from https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0
class Dataset(object):
    def __init__(self, generator=EntryGenerator())):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, entry_gen: EntryGenerator()):
        batch_size = 10
        prefetch_batch_buffer

        dataset = tf.data.Dataset.from_generator(entry_gen.get_next_entry, \
                                                 output_types={EntryGenerator.image: tf.Tensor, EntryGenerator.target: tf.Tensor, EntryGenerator.bbox_x1: tf.int64, EntryGenerator.bbox_y1: tf.int64, EntryGenerator.bbox_x2: tf.int64, EntryGenerator.bbox_y2: tf.int64})

        # dataset = dataset.map() - don't even need this
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)


def train_image(image_loader, images, tracker_trainer):
    """TODO: Docstring for train_image.
    """
    curr_image = np.random.randint(0, len(images))
    list_annotations = images[curr_image]
    curr_ann = np.random.randint(0, len(list_annotations))

    image, bbox = image_loader.load_annotation(curr_image, curr_ann)
    tracker_trainer.train(image, image, bbox, bbox)


def train_video(videos, tracker_trainer):
    """TODO: Docstring for train_video.
    """
    video_num = np.random.randint(0, len(videos))
    video = videos[video_num]

    annotations = video.annotations

    if len(annotations) < 2:
        logger.info('Error - video {} has only {} annotations', video.video_path, len(annotations))

    ann_index = np.random.randint(0, len(annotations) - 1)
    frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)

    frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)
    tracker_trainer.train(image_prev, image_curr, bbox_prev, bbox_curr)


def main(args):
    """TODO: Docstring for main.
    """

    logger.info('Loading training data')
    # Load imagenet training images and annotations
    # imagenet_folder = os.path.join(args['imagenet'], 'images')
    #imagenet_annotations_folder = os.path.join(args['imagenet'], 'gt')
    #objLoaderImgNet = loader_imagenet(imagenet_folder, imagenet_annotations_folder, logger)
    #train_imagenet_images = objLoaderImgNet.loaderImageNetDet()

    # Load alov training images and annotations
    alov_folder = os.path.join(args['alov'], 'images')
    alov_annotations_folder = os.path.join(args['alov'], 'gt')
    objLoaderAlov = loader_alov(alov_folder, alov_annotations_folder, logger)
    objLoaderAlov.loaderAlov()
    train_alov_videos = objLoaderAlov.get_videos()

    # create example generator and setup the network
    objExampleGen = example_generator(float(args['lamda_shift']), float(args['lamda_scale']), float(args['min_scale']), float(args['max_scale']), logger)
    objRegTrain = regressor_train(args['train_prototxt'], args['init_caffemodel'], int(args['gpu_id']), args['solver_prototxt'], logger)
    objTrackTrainer = tracker_trainer(objExampleGen, objRegTrain, logger)

    # NEW GOTURN PATH
    tracknet = goturn_net.TRACKNET(BATCH_SIZE)
    tracknet.build()

    # wrap the three lists together into a single object
    dataset = tf.data.Dataset.from_generator()

    global_step = tf.Variable(0, trainable=False, name="global_step")

    train_step = tf.train.AdamOptimizer(0.00001, 0.9).minimize( \
        tracknet.loss_wdecay, global_step=global_step)
    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train_summary', sess.graph)
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    start = 0
    if ckpt and ckpt.model_checkpoint_path:
        start = int(ckpt.model_checkpoint_path.split("-")[1])
        logging.info("start by iteration: %d" % (start))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
    assign_op = global_step.assign(start)
    sess.run(assign_op)
    model_saver = tf.train.Saver(max_to_keep=3)
    try:
        for i in range(start, int(len(train_box) / BATCH_SIZE * NUM_EPOCHS)):
            if i % int(len(train_box) / BATCH_SIZE) == 0:
                logging.info("start epoch[%d]" % (int(i / len(train_box) * BATCH_SIZE)))
                if i > start:
                    save_ckpt = "checkpoint.ckpt"
                    last_save_itr = i
                    model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)
            print(global_step.eval(session=sess))

            cur_batch = sess.run(batch_queue)

            start_time = time.time()
            [_, loss] = sess.run([train_step, tracknet.loss], feed_dict={tracknet.image: cur_batch[0],
                                                                         tracknet.target: cur_batch[1],
                                                                         tracknet.bbox: cur_batch[2]})
            logging.debug(
                'Train: time elapsed: %.3fs, average_loss: %f' % (time.time() - start_time, loss / BATCH_SIZE))

            if i % 10 == 0 and i > start:
                summary = sess.run(merged_summary, feed_dict={tracknet.image: cur_batch[0],
                                                              tracknet.target: cur_batch[1],
                                                              tracknet.bbox: cur_batch[2]})
                train_writer.add_summary(summary, i)
    except KeyboardInterrupt:
        print("get keyboard interrupt")
        if (i - start > 1000):
            model_saver = tf.train.Saver()
            save_ckpt = "checkpoint.ckpt"
            model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)
