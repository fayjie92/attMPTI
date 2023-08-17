""" Collect point clouds and the corresponding labels from original 3Frog dataset, and save into numpy files.

Author: Umamaheswaran Raman Kumar, 2023
"""

import os
import glob
import numpy as np
import pandas as pd
import sys
from pyntcloud import PyntCloud
#import pypcd.pypcd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='datasets/TFrog/TFrogDataset',
                        help='Directory to dataset')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    DST_PATH = os.path.join(ROOT_DIR, 'datasets/TFrog')
    SAVE_PATH = os.path.join(DST_PATH, 'scenes', 'data')
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    CLASS_NAMES = [x.rstrip().split()[0]
                   for x in open(os.path.join(ROOT_DIR, 'datasets/TFrog/meta', 'tfrog_classnames.txt'))]
    CLASS2COLOR = {x.rstrip().split()[0]:tuple(map(int, x.rstrip().split()[1:4]))
                   for x in open(os.path.join(ROOT_DIR, 'datasets/TFrog/meta', 'tfrog_classnames.txt')))}
    CLASS2LABEL = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    for folder in os.listdir(DATA_PATH):
        print('Processing folder : ', folder)
        for file in glob.glob(DATA_PATH+'/'+folder+'/converter/trainingdata/pcd/[!empty]*/*.obj.groundtruth.pcd'):
            filename = os.path.split(file)[1]
            print(filename)
            f_split = filename.split('.')
            out_filename = folder+'_'+ f_split[1]+'_'+f_split[0]+'.npy'
            out_filepath = os.path.join(SAVE_PATH, out_filename)
            point_cloud = PyntCloud.from_file(file)
            points = pd.DataFrame(point_cloud.points[['x','y','z','red','green','blue']])#.to_numpy()
            points['x'] = points['x'].div(100)
            points['y'] = points['y'].div(100)
            points['z'] = points['z'].div(100)
            points = points.assign(label=0)

            for key, (r,g,b) in CLASS2COLOR.items():
                label = CLASS2LABEL[key]
                points.loc[(points['red']==r) & (points['green']==g) & (points['blue']==b), 
                        'label'] = label
            points = points.to_numpy()

            file_format = 'numpy'
            if file_format == 'txt':
                fout = open(out_filepath, 'w')
                for i in range(points.shape[0]):
                    fout.write('%f %f %f %d %d %d %d\n' % \
                            (points[i, 0], points[i, 1], points[i, 2],
                                points[i, 3], points[i, 4], points[i, 5],
                                points[i, 6]))
                fout.close()
            elif file_format == 'numpy':
                np.save(out_filepath, points)
            else:
                print('ERROR!! Unknown file format: %s, please use txt or numpy.' % \
                    (file_format))
                exit()
