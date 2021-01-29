# --------------------------------------------------------------------------
# Tensorflow Implementation of Segmentation Fingerprint Generation
# Licensed under The MIT License [see LICENSE for details]
# Written by Young-Mook Kang
# Email: kym343@naver.com
# --------------------------------------------------------------------------

import logging
import utils as utils

class Fingerprint(object):
    def __init__(self, name='Fingerprint', track='Semantic_Segmentation_Dataset',
                 isTrain=True, resizedFactor=1.0, logDir=None):
        self.name = name
        self.track = track

        self.numTrainImgs = 700 # 8916
        self.numValImgs = 200   # 2403
        self.numTestImgs = 100  # 1440

        self.numTrainPersons = 70 # 95
        self.numValPersons = 20 # 28
        self.numTestPersons = 10 # 29
        self.numClasses = 2

        self.decodeImgShape = (int(320 * resizedFactor), int(280 * 2 * resizedFactor), 1)
        self.singleImgShape = (int(320 * resizedFactor), int(280 * resizedFactor), 1)

        # TFrecord path
        self.trainPath = '../../Data/Fingerprint/{}/train/train.tfrecords'.format(self.track)
        self.valPath = '../../Data/Fingerprint/{}/validation/validation.tfrecords'.format(self.track)
        self.testPath = '../../Data/Fingerprint/{}/test/test.tfrecords'.format(self.track)
        self.overfittingPath = '../../Data/Fingerprint/{}/overfitting/overfitting.tfrecords'.format(self.track)

        if isTrain:
            self.logger = logging.getLogger(__name__)   # logger
            self.logger.setLevel(logging.INFO)
            utils.init_logger(logger=self.logger, logDir=logDir, isTrain=isTrain, name='dataset')

            self.logger.info('Dataset name: \t\t{}'.format(self.name))
            self.logger.info('Dataset track: \t\t{}'.format(self.track))
            self.logger.info('Num. of training imgs: \t{}'.format(self.numTrainImgs))
            self.logger.info('Num. of validation imgs: \t{}'.format(self.numValImgs))
            self.logger.info('Num. of test imgs: \t\t{}'.format(self.numTestImgs))
            self.logger.info('Num. of training persons: \t{}'.format(self.numTrainPersons))
            self.logger.info('Num. of validation persons: \t{}'.format(self.numValPersons))
            self.logger.info('Num. of test persons: \t{}'.format(self.numTestPersons))
            self.logger.info('Num. of classes: \t\t{}'.format(self.numClasses))
            self.logger.info('Decode image shape: \t\t{}'.format(self.decodeImgShape))
            self.logger.info('Single img shape: \t\t{}'.format(self.singleImgShape))
            self.logger.info('Training TFrecord path: \t{}'.format(self.trainPath))
            self.logger.info('Validation TFrecord path: \t{}'.format(self.valPath))
            self.logger.info('Test TFrecord path: \t\t{}'.format(self.testPath))
            self.logger.info('Overfitting TFrecord path: \t\t{}'.format(self.overfittingPath))

    def __call__(self, isTrain=True):
        if isTrain:
            return self.trainPath, self.valPath, self.overfittingPath
        else:
            return self.testPath, self.valPath, None

def Dataset(name, track='Semantic_Segmentation_Dataset', isTrain=True, resizedFactor=1.0, logDir=None):
    if name == 'Fingerprint' and track == 'Semantic_Segmentation_Dataset':
        return Fingerprint(name=name, track=track, isTrain=isTrain, resizedFactor=resizedFactor, logDir=logDir)
    else:
        raise NotImplementedError
