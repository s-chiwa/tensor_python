from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import sys
import tarfile

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'model_dir', os.path.dirname(os.path.abspath(sys.argv[0])),
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

class NodeLookup(object):

	def __init__(self,label_lookup_path=None,uid_lookup_path=None):
		if not label_lookup_path:
			label_lookup_path = os.path.join(FLAGS.model_dir,'imagenet_2012_challenge_label_map_proto.pbtxt')
 		if not uid_lookup_path:
      			uid_lookup_path = os.path.join(FLAGS.model_dir,'imagenet_synset_to_human_label_map.txt')
			#uid_lookup_path = os.path.join(FLAGS.model_dir,'revision_human_label_map_jp.txt')
		self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

	def load(self, label_lookup_path, uid_lookup_path):
		if not tf.gfile.Exists(uid_lookup_path):
			tf.logging.fatal('File does not exist %s',uid_lookup_path)
		if not tf.gfile.Exists(label_lookup_path):
			tf.logging.fatal('File does not exist %s', label_lookup_path)

		proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
		uid_to_human = {}
		p = re.compile(r'[n\d]*[ \S,]*')
		for line in proto_as_ascii_lines:
			parsed_items = p.findall(line)
			uid = parsed_items[0]
			human_string = parsed_items[2]
			uid_to_human[uid] = human_string

		#print(uid_to_human)

		node_id_to_uid = {}
		proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
		for line in proto_as_ascii:
			#print(line)
			if line.startswith('  target_class:'):
				target_class = int(line.split(': ')[1])
			if line.startswith('  target_class_string:'):
				target_class_string = line.split(': ')[1]
				node_id_to_uid[target_class] = target_class_string[1:-2]

		#print(node_id_to_uid)

		node_id_to_name = {}
		for key, val in node_id_to_uid.items():
			if val not in uid_to_human:
				tf.logging.fatal('Failed to locate: %s', val)
			name = uid_to_human[val]
			node_id_to_name[key] = name

		return node_id_to_name

	def id_to_string(self, node_id):
		if node_id not in self.node_lookup:
			return ''
		return self.node_lookup[node_id]



class TensorClassifyImage:


	def create_graph(self):
		with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir,'classify_image_graph_def.pb'), 'rb') as f:
    			graph_def = tf.GraphDef()
    			graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(graph_def, name='')


	def run_inference_on_image(self,image):
		ans = ""

		imgPath = os.path.join(FLAGS.model_dir,image)

		print(imgPath+" is anlysising...")

		if not tf.gfile.Exists(imgPath):
	    		tf.logging.fatal('File does not exist %s', imgPath)
	  	image_data = tf.gfile.FastGFile(imgPath, 'rb').read()

		# Creates graph from saved GraphDef.
	  	self.create_graph()

		with tf.Session() as sess:
			softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
			predictions = sess.run(softmax_tensor,{'DecodeJpeg/contents:0': image_data})
			predictions = np.squeeze(predictions)

			node_lookup = NodeLookup()

			top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
			for node_id in top_k:
				human_string = node_lookup.id_to_string(node_id)
				score = predictions[node_id]
				ans = ans + '%s (score = %.5f)' % (human_string, score) + ":"

		return ans

