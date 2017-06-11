import caffe
import os
import numpy as np
import sys
import cv2
import csv
import glob
net = caffe.Net('/home/nimbix/caffe/models/bvlc_alexnet/deploy.prototxt',
'/home/nimbix/caffe/models/bvlc_alexnet/snapshots/iter_iter_30000.caffemodel.h5', caffe.TEST)
csv_file = open('/home/nimbix/data/explorers/filename_uuid_map_test_data.csv')
test_pngs = glob.glob('/home/nimbix/data/explorers/test_data_png/*.png')
file_f = open('/home/nimbix/data/explorers/result4.csv', 'w')
caffe.set_device(0)
caffe.set_mode_gpu()
for line in csv_file:
    line = line.strip('\n')
    file_info = line.split(',')
    img = cv2.imread(file_info[0], cv2.IMREAD_COLOR)
    img = img.astype(float)
    img -= np.array([125.0, 125.0, 125.0])
    img = img.transpose((2, 0, 1))
    net.blobs['data'].reshape(1, *img.shape)
    net.blobs['data'].data[...]= img
    output = net.forward()
    result = output['prob']
    writer = csv.writer(file_f)
    writer.writerow([file_info[1], result[0][3], result[0][2], result[0][0], result[0][1]])
    
