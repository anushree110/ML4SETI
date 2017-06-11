mport numpy as np
import lmdb
import caffe
file_path = ''
N = 1000
f = open(file_path)
map_size =  10e12

env = lmdb.open('train_lmdb', map_size=map_size)
count = 0
with env.begin(write=True) as txn:
   for line in f:
      line = line.strip('\n')
      file_info = line.strip(",")
      pdb.set_trace()
      img = cv2.imread(file_info[0], cv2.IMREAD_COLOR)
      pdb.set_trace()
      img = img.transpose((2, 1, 0))
      label = file_info[1]
      datum = caffe.proto.caffe_pb2.Datum()
      datum.channels = img.shape[0]
      datum.height = img.shape[1]
      datum.width = img.shape[2]
      datum.label = int(file_info[1])
      str_id = '{:08}'.format(count)
      count = count + 1
      txn.put(str_id.encode('ascii'), datum.SerializeToString())
