import sys
sys.path.append('/home/spark/caffe/caffe/python')

import caffe
from caffe.proto.caffe_pb2 import Datum

import os
os.environ['LMDB_FORCE_CFEI'] = '1'

import lmdb

feature_dimension = 123

def extract(filename, dbname):
	env = lmdb.open(dbname, map_size=2**30)
	with env.begin(write=True) as txn:
		with open(filename) as fin:
			for i, line in enumerate( fin.readlines() ):
				elem = line.strip().split(' ')
				d = Datum()
				if elem[0] == '+1':
					d.label = 1
				else:
					d.label = 0

				features = [0] * feature_dimension
				for e in elem[1:]:
					pos,v = e.split(':')
					features[int(pos) - 1] = 1
				d.channels = 1
				d.height = 1
				d.width = feature_dimension
				d.data = "".join([chr(x) for x in features])
				txn.put(str(i), d.SerializeToString())


extract('a1a.txt', '/home/spark/Desktop/a1a/train/')
extract('a1a_test.txt', '/home/spark/Desktop/a1a/test')