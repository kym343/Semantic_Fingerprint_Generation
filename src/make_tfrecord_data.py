# --------------------------------------------------------------------------
# Tensorflow Implementation of Segmentation Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------

import os
import numpy as np
import tensorflow as tf

import utils as utils


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('input_data', '../../Data/Fingerprint/Semantic_Segmentation_Dataset',
                       'data input directory, default: ../../Data/Fingerprint/Semantic_Segmentation_Dataset')
tf.flags.DEFINE_string('output_data', '../../Data/Fingerprint/Semantic_Segmentation_Dataset',
                       'data output directory, default: ../../Data/Fingerprint/Semantic_Segmentation_Dataset')
tf.flags.DEFINE_string('stage', 'train', 'stage selection from [train|validation|test|overfitting], default: train')


def data_writer(inputDir, stage, outputName):
    # dataPath = os.path.join(inputDir, '{}'.format(stage), 'paired')
    dataPath = os.path.join(inputDir, 'train', 'paired')
    imgPaths = utils.all_files_under(folder=dataPath, subfolder='')

    is_edit = False
    is_random = False
    if is_edit:
        num_class = 100
        num_sample_each_class = 10

        if is_random:
            num_of_train = 7
            val_rate = 0.2
            total_sample_num = np.array(range(num_sample_each_class))

            train_sample_num = np.random.choice(num_sample_each_class, num_of_train, replace=False)
            val_sample_num = np.random.choice(np.array(list(set(total_sample_num) - set(train_sample_num)))
                                              , int(num_sample_each_class * val_rate), replace=False)
            test_sample_num = np.array(list(set(total_sample_num) - set(train_sample_num) - set(val_sample_num)))
        else:
            train_sample_num = [1, 2, 5, 3, 0, 9, 4]
            val_sample_num = [7, 8]
            test_sample_num = [6]

        train_idx = list(
            cls * num_sample_each_class + num_ for cls in range(num_class) for num_ in train_sample_num)
        test_idx = list(
            cls * num_sample_each_class + num_ for cls in range(num_class) for num_ in test_sample_num)
        val_idx = list(
            cls * num_sample_each_class + num_ for cls in range(num_class) for num_ in val_sample_num)

        is_print = True
        if is_print:
            print("============================================")
            print("train_sample_num : {}".format(train_sample_num))
            print("val_sample_num : {}".format(val_sample_num))
            print("test_sample_num : {}".format(test_sample_num))
            print("============================================")

        temp = []
        if stage == 'train':
            temp = list(imgPaths[i] for i in train_idx)
        elif stage == 'validation':
            temp = list(imgPaths[i] for i in val_idx)
        elif stage == 'test':
            temp = list(imgPaths[i] for i in test_idx)

        imgPaths = temp # train_idx | test_idx | val_idx

    numImgs = len(imgPaths)

    # Create tfrecrods dir if not exists
    output_file = '{0}/{1}/{1}.tfrecords'.format(outputName, stage)
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # Dump to tfrecords file
    writer = tf.io.TFRecordWriter(output_file)

    for idx, img_path in enumerate(imgPaths):
        with tf.io.gfile.GFile(img_path, 'rb') as f:
            img_data = f.read()

        example = _convert_to_example(img_path, img_data)
        writer.write(example.SerializeToString())

        if np.mod(idx, 100) == 0:
            print('Processed {}/{}...'.format(idx, numImgs))

    print('Finished!')
    writer.close()


def _convert_to_example(imgPath, imgBuffer):
    # Build an example proto
    imgName = os.path.basename(imgPath)
    userId = imgName.replace('.png', '').split('_')[-1]

    example = tf.train.Example(features=tf.train.Features(
        feature={'image/file_name': _bytes_feature(tf.compat.as_bytes(imgName)),
                 'image/user_id': _bytes_feature(tf.compat.as_bytes(userId)),
                 'image/encoded_image': _bytes_feature(imgBuffer)}))

    return example


def _bytes_feature(value):
    # Wrapper for inserting bytes features into example proto
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    print("Convert {} - {} data to tfrecrods...".format(FLAGS.input_data, FLAGS.stage))
    data_writer(FLAGS.input_data, FLAGS.stage, FLAGS.output_data)


if __name__ == '__main__':
    tf.compat.v1.app.run()
