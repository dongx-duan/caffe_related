from __future__ import print_function

import sys
sys.path.append("/home/hadoop/caffe/python")
import caffe
import skimage
import numpy as np

VGG_MEAN = [103.939, 116.779, 123.68]

def load_image(path):
  # load image
  img = skimage.io.imread(path)
  img = img / 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  #print "Original Image Shape: ", img.shape
  # we crop image from center
  short_edge = min(img.shape[:2])
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to 224, 224
  resized_img = skimage.transform.resize(crop_img, (224, 224))
  return resized_img

def preprocess(img):
  out = np.copy(img) * 255
  out = out[:, :, [2,1,0]] # swap channel from RGB to BGR
  # sub mean
  out[:,:,0] -= VGG_MEAN[0]
  out[:,:,1] -= VGG_MEAN[1]
  out[:,:,2] -= VGG_MEAN[2]
  out = out.transpose((2,0,1)) # h, w, c -> c, h, w
  return out

caffe.set_mode_gpu()
net_caffe = caffe.Net("VGG_2014_16.prototxt", "VGG_ILSVRC_16_layers.caffemodel", caffe.TEST)
labels = np.loadtxt('synset.txt', str, delimiter='\t')
using_debug_mode = False

def forward(image_url):
	print("load_image: ", image_url)
	net_caffe.blobs['data'].data[0] = preprocess(load_image(image_url))
	print("Caffe Net forward()", image_url)
	net_caffe.forward()

import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):
    	self.write("Extract Image Features from VGG16")

class ExtractHander(tornado.web.RequestHandler):
	def get(self, image_url):
		forward(image_url)
		fc6 = net_caffe.blobs['fc6'].data[0]
		assert(len(fc6) == 4096)
		output = fc6.tostring()
		if using_debug_mode:
			assert( np.array_equal(fc6, np.fromstring(output, dtype=np.float32)))
		self.write(output)

class ObjectRecognizeHandler(tornado.web.RequestHandler):
	def get(self, image_url):
		forward(image_url)
		topk = net_caffe.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
		res = [a for a in labels[topk]]
		self.write( str(res) )

if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/feature/(.*)", ExtractHander),
        (r"/object/(.*)", ObjectRecognizeHandler),
    ])
    port = 8888
    application.listen(port)
    print("server listen on ", port)
    tornado.ioloop.IOLoop.current().start()


# start server
# python caffe_feature.py
# open a browser
# http://9.186.106.72:8888/feature/https://www.petfinder.com/wp-content/uploads/2012/11/140272627-grooming-needs-senior-cat-632x475.jpg
# 