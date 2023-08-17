""" Data Preprocess and Loader for 3Frog Dataset

Author: Zhao Na, 2020
Modified: Umamaheswaran Raman Kumar, 2023
"""
import os
import sys
import glob
import numpy as np
import pickle

# get the logger working
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from utils.log import logger_log
logger = logger_log()

class TFrogDataset(object):
    def __init__(self, cvfold, data_path):
        self.data_path = data_path
        self.classes = 7
        # self.class2type = {0:'wall', 1:'floor', 2:'ceiling', 3:'door', 4:'doorframe', 
        #                    5:'window', 6:'clutter'}
        class_names = [x.rstrip().split()[0]
                   for x in open(os.path.join(os.path.dirname(data_path), 'meta', '3frog_classnames.txt'))]
        self.class2type = {i: name.strip() for i, name in enumerate(class_names)}
        logger.info(f'classes mapping: {self.class2type}')

        self.type2class = {self.class2type[t]: t for t in self.class2type}
        self.types = self.type2class.keys()
        self.fold_0 = ['wall', 'ceiling', 'doorframe', 'clutter']
        self.fold_1 = ['floor', 'door', 'window']

        #logger.info(f'fold_0 classes: {self.fold_0}')
        #logger.info(f'fold_1 classes: {self.fold_1}')

        if cvfold == 0:
            self.test_classes = [self.type2class[i] for i in self.fold_0]
        elif cvfold == 1:
            self.test_classes = [self.type2class[i] for i in self.fold_1]
        else:
            raise NotImplementedError('Unknown cvfold (%s). [Options: 0,1]' %cvfold)

        all_classes = [i for i in range(0, self.classes-1)]
        self.train_classes = [c for c in all_classes if c not in self.test_classes]

        # print('train_class:{0}'.format(self.train_classes))
        # print('test_class:{0}'.format(self.test_classes))

        self.class2scans = self.get_class2scans()

    def get_class2scans(self):
        class2scans_file = os.path.join(self.data_path, 'class2scans.pkl')
        if os.path.exists(class2scans_file):
            #load class2scans (dictionary)
            with open(class2scans_file, 'rb') as f:
                class2scans = pickle.load(f)
        else:
            min_ratio = .05  # to filter out scans with only rare labelled points
            min_pts = 100  # to filter out scans with only rare labelled points
            class2scans = {k:[] for k in range(self.classes)}

            for file in glob.glob(os.path.join(self.data_path, 'processed/scenes/blocks/', '*.npy')):
                scan_name = os.path.basename(file)[:-4]
                data = np.load(file)
                labels = data[:,6].astype(int)
                classes = np.unique(labels)
                print('{0} | shape: {1} | classes: {2}'.format(scan_name, data.shape, list(classes)))
                for class_id in classes:
                    #if the number of points for the target class is too few, do not add this sample into the dictionary
                    num_points = np.count_nonzero(labels == class_id)
                    threshold = max(int(data.shape[0]*min_ratio), min_pts)
                    if num_points > threshold:
                        class2scans[class_id].append(scan_name)

            print('==== class to scans mapping is done ====')
            for class_id in range(self.classes):
                print('\t class_id: {0} | min_ratio: {1} | min_pts: {2} | class_name: {3} | num of scans: {4}'.format(
                          class_id,  min_ratio, min_pts, self.class2type[class_id], len(class2scans[class_id])))

            with open(class2scans_file, 'wb') as f:
                pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)
        return class2scans