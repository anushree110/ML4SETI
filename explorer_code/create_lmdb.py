import numpy as np
import pdb
import lmdb
import caffe
import cv2
from PIL import Image
file_path = '/home/nimbix/data/explorers/valid_class_label_map.csv'
N = 1000
f = open(file_path)
map_size =  10e12

env = lmdb.open('val_lmdb', map_size=map_size)
count = 0
with env.begin(write=True) as txn:
   for line in f:
      line = line.strip('\n')
      file_info = line.split(",")
      img = cv2.imread(file_info[0], cv2.IMREAD_COLOR)
      #img = np.array(Image.open(file_info[0]).convert('RGB'))
      bgr_img = img.transpose((2, 0, 1))
      label = file_info[1]
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = bgr_img.shape[0]
      datum.height = bgr_img.shape[1]
      datum.width = bgr_img.shape[2]
      datum.data = bgr_img.tobytes()
      datum.label = int(file_info[1]) 
      str_id = '{:08}'.format(count)
      count = count + 1
      txn.put(str_id.encode('ascii'), datum.SerializeToString())
