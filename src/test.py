# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob
import math
import struct
from sklearn.cluster import KMeans




import numpy as np
import tensorflow as tf




from config import *
from train import _draw_box
from nets import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-535000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")



def prune_edges_with_small_weight(ndarray,percent):
    weights = ndarray.flatten()
    abso = np.absolute(weights)
    threshold = np.sort(abso)[int(math.ceil(weights.size * percent / 100))]
    weights[abso < threshold] = 0
    return weights.reshape(ndarray.shape)


def relative_index(absolute_index, ndarray, max_index):
    first = absolute_index[0]
    relative = np.insert(np.diff(absolute_index), 0, first)
    dense = ndarray.tolist()
    max_index_or_less = relative.tolist()
    shift = 0
    
    for i in np.where(relative > max_index)[0].tolist():
        while max_index_or_less[i+shift] > max_index:
            max_index_or_less.insert(i+shift,max_index)
            dense.insert(i+shift,0)
            shift+=1
            max_index_or_less[i+shift] -= max_index
    
    return (np.array(max_index_or_less), np.array(dense))

def store_compressed_network(path, layers):
    
    if os.path.exists(path):
        return path
    with open(path, 'wb') as f:
        for layer in layers:
            f.write(struct.pack('Q',layer[1].size))
            f.write(layer[0].tobytes())
            f.write(layer[1].tobytes())
            f.write(struct.pack('Q',layer[2].size))
            f.write(layer[2].tobytes())
            f.write(layer[3].tobytes())
    return path

def compress(pruning_percent, cluster_num):
  """Detect image."""

  assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
      'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

  with tf.Graph().as_default():
    # Load model
    if FLAGS.demo_net == 'squeezeDet':
      mc = kitti_squeezeDet_config()
      mc.BATCH_SIZE = 1
      # model parameters will be restored from checkpoint
      mc.LOAD_PRETRAINED_MODEL = False
      model = SqueezeDet(mc, FLAGS.gpu)
    
    saver = tf.train.Saver(model.model_params)
    components=[]
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, FLAGS.checkpoint)
        for l in model.model_params:
            if 'kernel' in l.name:
                w = sess.run(l)
                #pruning
                sparse_1d = prune_edges_with_small_weight(w, pruning_percent).flatten()
                #K-means
                nonzero = sparse_1d[sparse_1d!=0]
                clusters = KMeans(n_clusters=cluster_num).fit(nonzero.reshape(-1,1).astype(np.float64))
                
                relative_index_in_4bits, cluster_labels = relative_index(np.where(sparse_1d !=0)[0],clusters.labels_+1,max_index=16-1)
                
                if relative_index_in_4bits.size % 2 == 1:
                    relative_index_in_4bits = np.append(relative_index_in_4bits,0)
              
                pair_of_4bits_in_1byte = relative_index_in_4bits[np.arange(0, relative_index_in_4bits.size, 2)] * 16 + relative_index_in_4bits[np.arange(1, relative_index_in_4bits.size, 2)]
              
                components.append([pair_of_4bits_in_1byte.astype(np.dtype('u1')), cluster_labels.astype(np.dtype('u1')), clusters.cluster_centers_.astype(np.float32).flatten()])  
            else:
                b = sess.run(l)
                components[-1].append(b)
                

            #op = tf.assign(l,w, validate_shape=False)
            #sess.run(op)
            #print (l.name)        
        #w = sess.run(l)
        #saver.save(sess,FLAGS.checkpoint) 
      #return
        path = store_compressed_network(str(pruning_percent)+'%_'+str(cluster_num)+'_cluster.npy', components)        
        return path
  
        for f in glob.iglob(FLAGS.input_path):
            im = cv2.imread(f)
            im = im.astype(np.float32, copy=False)
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
            input_image = im - mc.BGR_MEANS

            # Detect
            det_boxes, det_probs, det_class = sess.run(
                [model.det_boxes, model.det_probs, model.det_class],
                feed_dict={model.image_input:[input_image]})

            # Filter
            final_boxes, final_probs, final_class = model.filter_prediction(
                det_boxes[0], det_probs[0], det_class[0])

            keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
            final_boxes = [final_boxes[idx] for idx in keep_idx]
            final_probs = [final_probs[idx] for idx in keep_idx]
            final_class = [final_class[idx] for idx in keep_idx]

            # TODO(bichen): move this color dict to configuration file
            cls2clr = {
                'car': (255, 191, 0),
                'cyclist': (0, 191, 255),
                'pedestrian':(255, 0, 191)
            }

            # Draw boxes
            _draw_box(
                im, final_boxes,
                [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                    for idx, prob in zip(final_class, final_probs)],
                cdict=cls2clr,
            )

            file_name = os.path.split(f)[1]
            out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
            cv2.imwrite(out_file_name, im)
            print ('Image detection output saved to {}'.format(out_file_name))

def decode(compressed_network_path):

    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.BATCH_SIZE = 1
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            model = SqueezeDet(mc, FLAGS.gpu)
        
        f = open(compressed_network_path, 'rb')

        saver = tf.train.Saver(model.model_params)
        components=[]
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            #saver.restore(sess, FLAGS.checkpoint)
            for l in model.model_params:
                if 'kernel' in l.name:
                    edge_num=np.fromfile(f,dtype=np.int64,count=1)
                    indices_pair_of_4bits = np.fromfile(f,dtype=np.dtype('u1'),count=int(math.ceil(edge_num[0]/2.0)))
                    cluster_labels=np.fromfile(f,dtype=np.dtype('u1'),count=edge_num)
                    clusters_num=np.fromfile(f,dtype=np.int64,count=1)
                    sharing_weights=np.fromfile(f,dtype=np.float32,count=clusters_num)
                
                    relative=np.zeros(indices_pair_of_4bits.size*2,dtype=np.dtype('u1'))
                    relative[np.arange(0,relative.size,2)] = indices_pair_of_4bits / 16
                    relative[np.arange(1,relative.size,2)] = indices_pair_of_4bits % 16

                    if relative[-1] == 0:
                        relative = relative[:-1]
                    
                    weights = np.zeros(np.prod(l.get_shape()),np.float32)
                    index = np.cumsum(relative)
                    weights[index] = np.insert(sharing_weights,0,0)[cluster_labels]

                    weights = weights.reshape(l.get_shape())
                    op = tf.assign(l,weights, validate_shape=False)
                    sess.run(op)

                else:
                    size = np.prod(l.get_shape())
                    biases=np.fromfile(f,dtype=np.float32,count=size)
                    op = tf.assign(l,biases, validate_shape=False)
                    sess.run(op)
            remain = f.read()
            f.close()
            if len(remain) != 0:
                sys.exit("Decode error!")

            saver.save(sess, 'model_decode_10_128.ckpt')
                


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    if FLAGS.mode == 'image':
        path=compress(10,128)
        
        #path = '30%_64_cluster.npy'

        decode(path) 
        
        print (path)
    else:
        video_demo()

if __name__ == '__main__':
    tf.app.run()
