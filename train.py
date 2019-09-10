# train file

import logging
import time
import tensorflow as tf
import os
import goturn_net

NUM_EPOCHS = 500
BATCH_SIZE = 1
WIDTH = 128
HEIGHT = 128
train_txt = "train_set.txt"
logfile = "train.log"

if __name__ == "__main__":
    if (os.path.isfile(logfile)):
        os.remove(logfile)
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.DEBUG, filename=logfile)

    # connect to
    target_tensors = tf.convert_to_tensor(train_target, dtype=tf.string)
    search_tensors = tf.convert_to_tensor(train_search, dtype=tf.string)
    box_tensors = tf.convert_to_tensor(train_box, dtype=tf.float64)
    input_queue = tf.data.Dataset.from_tensor_slices([search_tensors, target_tensors, box_tensors]).shuffle=(tf.shape()))
    batch_queue = next_batch(input_queue)
    tracknet = goturn_net.TRACKNET(BATCH_SIZE)
    tracknet.build()

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

    coord = tf.train.Coordinator()
    # start the threads
    tf.train.start_queue_runners(sess=sess, coord=coord)

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
