# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Simple transfer learning with Inception v3 or Mobilenet models.

With support for TensorBoard.

This example shows how to take a Inception v3 or Mobilenet model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector (1001-dimensional for
Mobilenet) for each image. We train a softmax layer on top of this
representation. Assuming the softmax layer contains N labels, this corresponds
to learning N + 2048*N (or 1001*N)  model parameters corresponding to the
learned biases and weights.

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/flower_photos/daisy/photo1.jpg
~/flower_photos/daisy/photo2.jpg
...
~/flower_photos/rose/anotherphoto77.jpg
...
~/flower_photos/sunflower/somepicture.jpg

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:


```bash
bazel build tensorflow/examples/image_retraining:retrain && \
bazel-bin/tensorflow/examples/image_retraining/retrain \
    --image_dir ~/flower_photos
```

Or, if you have a pip installation of tensorflow, `retrain.py` can be run
without bazel:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos
```

You can replace the image_dir argument with any folder containing subfolders of
images. The label for each image is taken from the name of the subfolder it's
in.

This produces a new model file that can be loaded and run by any TensorFlow
program, for example the label_image sample code.

By default this script will use the high accuracy, but comparatively large and
slow Inception v3 model architecture. It's recommended that you start with this
to validate that you have gathered good training data, but if you want to deploy
on resource-limited platforms, you can try the `--architecture` flag with a
Mobilenet model. For example:

```bash
python tensorflow/examples/image_retraining/retrain.py \
    --image_dir ~/flower_photos --architecture mobilenet_1.0_224
```

There are 32 different Mobilenet models to choose from, with a variety of file
size and latency options. The first number can be '1.0', '0.75', '0.50', or
'0.25' to control the size, and the second controls the input image size, either
'224', '192', '160', or '128', with smaller sizes running faster. See
https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
for more information on Mobilenet.

To use with TensorBoard:

By default, this script will log summaries under /tf_files/training_summaries/ directory

Visualize the summaries with this command:

tensorboard --logdir /tf_files/training_summaries/[model_name]

To Do: (Diane)
(1) Resume training with weights from earilier models 
(2) Adaptive learning rate 

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
from datetime import datetime
import hashlib
import os.path
import random
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import os
import glob
from math import pi
import shutil
from pathlib import Path

# import skimage.io
# import skimage.transform
# from memory_profiler import profile
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# import constant

FLAGS = None

img_index = 0  # starting index of augmented image
# These are all parameters that are tied to the particular model architecture
# we're using for Inception v3. These include things like tensor names and their
# sizes. If you want to adapt this script to work with another model, you will
# need to update these to reflect the values in the network you're using.
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

LEARNING_RATE_REDUCTION_RATIO = 0.9 #0.3 #1 #0.1
BEST_RESULT_COUNT_FOR_LEARNING_RATE_REDUCTION = 100 #15 #50


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result = collections.OrderedDict()
  sub_dirs = [
    os.path.join(image_dir,item)
    for item in gfile.ListDirectory(image_dir)]
  sub_dirs = sorted(item for item in sub_dirs
                    if gfile.IsDirectory(item))
  for sub_dir in sub_dirs:
    extensions = ['jpg', 'jpeg', 'png', 'bmp'] #['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    tf.logging.info("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      tf.logging.warning('No files found')
      continue
    if len(file_list) < 20:
      tf.logging.warning(
          'WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      tf.logging.warning(
          'WARNING: Folder {} has more than {} images. Some images will '
          'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.basename(file_name)

      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      if FLAGS.hash_full_path == 'True':
        input_hash_name = file_name  # "/home/research/Public/Datasets/pcb/v2.1_LED_3gray_224_wo_anti_aliasing/bad/S2_G_774_i_356.png"
      else:
        input_hash_name = base_name  # "S2_G_774_i_356.png"

      if FLAGS.proper_data_partition == 'True':
        # hash_name = re.sub(r'_nohash_.*$', '', file_name)
        # hash_name = re.sub(r'_theta_.*$', '', file_name)
        hash_name = re.sub(r'_i_.*$', '', input_hash_name)
      else:
        hash_name = input_hash_name

      hash_name = hash_name + FLAGS.append_filename #### variate file_name to change file partition (training, validation, test sets) -- Di added !!!! the appended string can be changed to shuffle files

      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category, architecture):
  """"Returns a path to a bottleneck file for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    category: Name string of set to pull images from - training, testing, or
    validation.
    architecture: The name of the model architecture.

  Returns:
    File system path string to an image that meets the requested parameters.
  """
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '_' + architecture + '.txt'


def convert_2D_tensor_to_scalar(bottleneck_tensor):
  """
  Reduce the tensor spatial size with pooling (before input to classifier).
  For example, input tensor if of shape (1,14,14,512), the output tensor is of shape (1,1,1,512). 
  The 2 dimensions in the middle are reduced to a scalar.  
  """
  flattened_bottleneck_tensor = bottleneck_tensor
  # if need pooling 
  if len(bottleneck_tensor.shape) == 4 and bottleneck_tensor.shape[1] != 1:  
    pooling_dims = bottleneck_tensor.shape[1:3] # automatically detect the pooling operator size
    with tf.name_scope('final_pooling'):
      flattened_bottleneck_tensor = tf.layers.average_pooling2d(bottleneck_tensor, pool_size=pooling_dims, strides=(1, 1))
  return flattened_bottleneck_tensor

def squeeze_4D_tensor_to_2D(tensor):
  """
  Reduce rank of tensor [batch, height, width, channel]. If height & width equal to 1, remove these 2 dimensions.
  For example, input tensor is of shape (N,1,1,1024), output tensor is of shape (N, 1024) 
  """
  if len(tensor.shape) == 4:
      if tensor.shape[1] == 1 and tensor.shape[2] == 1:
          squeezed_tensor = tf.squeeze(tensor, axis=[1, 2])
  return squeezed_tensor

def create_model_graph(model_info):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Args:
    model_info: Dictionary containing information about the model architecture.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(FLAGS.pretrain_model_dir, model_info['model_file_name'])
    with tf.gfile.GFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              model_info['bottleneck_tensor_name'],
              model_info['resized_input_tensor_name'],
          ]))
      # if FLAGS.feature_vector == 'second_last' or FLAGS.feature_vector == 'L-2':
      #   bottleneck_tensor = tf.squeeze(bottleneck_tensor, [1, 2])  # THIS IS ADDED TO REDUCE DATA RANK, AFTER REMOVING THE LAST DENSE LAYER !!!!!!!! CONVERT BOTTLENECK_TENSOR FROM [1,1,1,1024] TO [1,1024]
      # elif FLAGS.feature_vector == 'layer7' or FLAGS.feature_vector == 'L7':  # mobilenet V1
      #   bottleneck_tensor = tf.layers.average_pooling2d(bottleneck_tensor, pool_size=(14, 14), strides=(1,1) )
      #   bottleneck_tensor = tf.squeeze(bottleneck_tensor, [1, 2])
      # elif FLAGS.feature_vector == 'layer6' or FLAGS.feature_vector == 'L6':  # mobilenet V2
      #   bottleneck_tensor = tf.layers.average_pooling2d(bottleneck_tensor, pool_size=(14, 14), strides=(1,1) )
      #   bottleneck_tensor = tf.squeeze(bottleneck_tensor, [1, 2])

      # The following 2 lines replaces the if conditions used previously 
      # Convert a matrix to a scalar. (1,14,14,512) ==> (1,1,1,512) 
      bottleneck_tensor = convert_2D_tensor_to_scalar(bottleneck_tensor)
      # Reduce rank. (1,1,1,512) ==> (1,512)
      bottleneck_tensor = squeeze_4D_tensor_to_2D(bottleneck_tensor)

  return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    decoded_image_tensor: Output of initial image resizing and  preprocessing.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  # First decode the JPEG image, resize it, and rescale the pixel values.
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  # Then run it through the recognition network.
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def maybe_download_and_extract(data_url):
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.

  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
  dest_directory = FLAGS.pretrain_model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded', filename, statinfo.st_size,
                    'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
  """Makes sure the folder exists on disk.

  Args:
    dir_name: Path string to the folder we want to create.
  """
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """Create a single bottleneck file."""
  tf.logging.info('Creating bottleneck at ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('File does not exist %s', image_path)
  image_data = tf.gfile.GFile(image_path, 'rb').read() # gfile.FastGFile
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('Error during processing file %s (%s)' % (image_path,
                                                                 str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor, architecture):
  """Retrieves or calculates bottleneck values for an image.

  If a cached version of the bottleneck data exists on-disk, return that,
  otherwise calculate the data and save it to disk for future use.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Integer offset of the image we want. This will be modulo-ed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string  of the subfolders containing the training
    images.
    category: Name string of which  set to pull images from - training, testing,
    or validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: The tensor to feed loaded jpeg data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The output tensor for the bottleneck values.
    architecture: The name of the model architecture.

  Returns:
    Numpy array of values produced by the bottleneck layer for the image.
  """
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category, architecture)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor, architecture):
  """Ensures all the training, testing, and validation bottlenecks are cached.

  Because we're likely to read the same image multiple times (if there are no
  distortions applied during training) it can speed things up a lot if we
  calculate the bottleneck layer values once for each image during
  preprocessing, and then just read those cached values repeatedly during
  training. Here we go through all the images we've found, calculate those
  values, and save them off.

  Args:
    sess: The current active TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    image_dir: Root folder string of the subfolders containing the training
    images.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    jpeg_data_tensor: Input tensor for jpeg data from file.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The penultimate output layer of the graph.
    architecture: The name of the model architecture.

  Returns:
    Nothing.
  """
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    # if FLAGS.update_bottleneck_file == 'True':  # remove cached bottleneck files
    #   label_lists = image_lists[label_name]
    #   sub_dir = label_lists['dir']
    #   sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    #   print('bottleneck path to remove:', sub_dir_path)
    #   if os.path.exists(sub_dir_path):
    #     shutil.rmtree(sub_dir_path)

    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)

        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor, architecture):
  """Retrieves bottleneck values for cached images.

  If no distortions are being applied, this function can retrieve the cached
  bottleneck values directly from disk for images. It picks a random set of
  images from the specified category.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: If positive, a random sample of this size will be chosen.
    If negative, all bottlenecks will be retrieved.
    category: Name string of which set to pull from - training, testing, or
    validation.
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    image_dir: Root folder string of the subfolders containing the training
    images.
    jpeg_data_tensor: The layer to feed jpeg image data into.
    decoded_image_tensor: The output of decoding and resizing the image.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.
    architecture: The name of the model architecture.

  Returns:
    List of bottleneck arrays, their corresponding ground truths, and the
    relevant filenames.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor, architecture)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor, architecture)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
    sess, image_lists, how_many, category, image_dir, input_jpeg_tensor, augmented_image_roi_tensor,
    distorted_image, resized_input_tensor, bottleneck_tensor, training_step_index):
  """Retrieves bottleneck values for training images, after distortions.

  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.

  Args:
    sess: Current TensorFlow Session.
    image_lists: Dictionary of training images for each label.
    how_many: The integer number of bottleneck values to return.
    category: Name string of which set of images to fetch - training, testing,
    or validation.
    image_dir: Root folder string of the subfolders containing the training
    images.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
    resized_input_tensor: The input node of the recognition graph.
    bottleneck_tensor: The bottleneck output layer of the CNN graph.

  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  for unused_i in range(how_many):
    label_index = random.randrange(class_count)
    label_name = list(image_lists.keys())[label_index]
    image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
    image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                category)
    if not gfile.Exists(image_path):
      tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = tf.gfile.GFile(image_path, 'rb').read()
    
    # Note that we materialize the distorted_image_data as a numpy array before
    # sending running inference on the image. This involves 2 memory copies and
    # might be optimized in other implementations.
    # distorted_image_data = sess.run(distorted_image,
    #                                 {input_jpeg_tensor: jpeg_data})

    # Get both augmented_image (augmented_image_data), and normalized_augmented_image (i.e., distorted_image_data)      
    distorted_image_data, augmented_image_data = sess.run([distorted_image, augmented_image_roi_tensor],
                                    {input_jpeg_tensor: jpeg_data})
    # save augmented image data as image for visualization
    if FLAGS.visualize_augmented_data == 'True': 
      # print('--------------',augmented_image_data.shape)

      # # Clear out any prior log data.
      # # !rm -rf logs
      # # Sets up a timestamped log directory.
      # logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
      # # Creates a file writer for the log directory.
      # file_writer = tf.summary.create_file_writer(logdir)
      # # Using the file writer, log the reshaped image.
      # with file_writer.as_default():
      #   # tf.summary.image("Training data", img, step=0)
      #   tf.summary.image('augmented_image', augmented_image_data, max_outputs=1)

      if training_step_index < 10: # save augmented image in the first 10 iterations
        augmented_image_name = './augmented_images/iter%s_ind%s.jpg' % (training_step_index, unused_i)
        save_tensor_as_image(augmented_image_data, augmented_image_name) 


    del augmented_image_data

    bottleneck_values = sess.run(bottleneck_tensor,
                                 {resized_input_tensor: distorted_image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck_values)
    ground_truths.append(ground_truth)
  return bottlenecks, ground_truths


def should_distort_images(flip_left_right, flip_up_down, random_crop, random_scale,
                          random_brightness, random_rotate, random_shift_delta, roi_param):
  """Whether any distortions are enabled, from the input flags.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.

  Returns:
    Boolean value indicating whether any distortions should be applied.
  """
  return (flip_left_right or flip_up_down or (random_crop != 0) or (random_scale != 0) or
          (random_brightness != 0) or (random_rotate != 0) or (random_shift_delta != 0) or
          (roi_param['roi_offset_x'] != 0) or (roi_param['roi_offset_y'] != 0) or (roi_param['roi_width'] != -1) or (roi_param['roi_height'] != -1) )


def add_input_distortions(flip_left_right, flip_up_down, random_crop, random_scale,
                          random_brightness, random_rotate, input_width, input_height,
                          input_depth, input_mean, input_std, roi_param, random_shift_delta, zoom_in):
  # """Creates the operations to apply the specified distortions.
  # Args:
  #   flip_left_right: Boolean whether to randomly mirror images horizontally.
  #   random_crop: Integer percentage setting the total margin used around the
  #   crop box.
  #   random_scale: Integer percentage of how much to vary the scale by.
  #   random_brightness: Integer range to randomly multiply the pixel values by.
  #   graph.
  #   input_width: Horizontal size of expected input image to model.
  #   input_height: Vertical size of expected input image to model.
  #   input_depth: How many channels the expected input image should have.
  #   input_mean: Pixel value that should be zero in the image for the graph.
  #   input_std: How much to divide the pixel values by before recognition.

  # Returns:
  #   The jpeg input layer and the distorted result tensor.
  #   augmented_image: 4-D tensor
  #   distort_result: 4-D tensor 
  # """

  # Augment image
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  augmented_image = image_augmentation(jpeg_data, flip_left_right, flip_up_down, random_crop, random_scale,
                          random_brightness, random_rotate, input_width, input_height,
                          input_depth, roi_param, random_shift_delta, zoom_in)  
  # tf.summary.image('augmented_image', augmented_image, max_outputs=1)

  # Resize image 
  resized = input_image_resize(augmented_image, input_height, input_width)

  # Normalize image  
  distort_result = input_image_normalization(resized, input_mean, input_std)
  return jpeg_data, augmented_image, distort_result

def image_augmentation(jpeg_data, flip_left_right, flip_up_down, random_crop, random_scale,
                          random_brightness, random_rotate, input_width, input_height,
                          input_depth, roi_param, random_shift_delta, zoom_in):
  """Creates the operations to apply the specified distortions.

  During training it can help to improve the results if we run the images
  through simple distortions like crops, scales, and flips. These reflect the
  kind of variations we expect in the real world, and so can help train the
  model to cope with natural data more effectively. Here we take the supplied
  parameters and construct a network of operations to apply them to an image.

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    graph.
    input_width: Horizontal size of expected input image to model.
    input_height: Vertical size of expected input image to model.
    input_depth: How many channels the expected input image should have.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    The jpeg input layer and the distorted result tensor.
  """

  # jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_image(jpeg_data, channels=input_depth) # changed to use the more generic decode_image function. # decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
   
  # "augment_random_rotate" requires images to have a known rank. We use a resizing function to force the tensor rank to be known.
  original_input_shape = [1080, 1440]
  # Input of resize_bilinear is 4-D with shape [batch, height, width, channels]
  decoded_image_4d = tf.image. resize_nearest_neighbor(decoded_image_4d,
                                              original_input_shape)
  # print('precropped_image RANK +++++++++++++', precropped_image.get_shape().ndims)
  preprocessed_image_3d = tf.squeeze(decoded_image_4d, squeeze_dims=[0])


  # margin_scale = 1.0 + (random_crop / 100.0)
  # resize_scale = 1.0 + (random_scale / 100.0)
  # margin_scale_value = tf.constant(margin_scale)
  # resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
  #                                        minval=1.0,
  #                                        maxval=resize_scale)
  # scale_value = tf.multiply(margin_scale_value, resize_scale_value)
  # precrop_width = tf.multiply(scale_value, input_width)
  # precrop_height = tf.multiply(scale_value, input_height)
  # precrop_shape = tf.stack([precrop_height, precrop_width])
  # precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  # precropped_image = tf.image.resize_bilinear(decoded_image_4d,
  #                                             precrop_shape_as_int)
  # print('precropped_image RANK +++++++++++++', precropped_image.get_shape().ndims)
  # precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  # print('precropped_image_3d RANK +++++++++++++', precropped_image_3d.get_shape().ndims)
  # cropped_image = tf.random_crop(precropped_image_3d,
  #                                [input_height, input_width, input_depth])


  augmented_image = augment_flip(preprocessed_image_3d, flip_left_right, flip_up_down)

  augmented_image = augment_brightness(augmented_image, random_brightness)

  augmented_image = augment_random_rotate(augmented_image, random_rotate)

  augmented_image = random_shift_and_crop(augmented_image, roi_param, random_shift_delta)

  augmented_image = zoom_central(augmented_image, zoom_in)

  # convert tensor from rank 3 to rank 4
  augmented_image = tf.expand_dims(augmented_image, 0)


  return augmented_image

def random_shift_and_crop(image, roi_param, random_shift_delta):
  '''Randomly shift the ROI by "random_shift_delta" pixels. Then crop the roi.
     This is implemented as (1) extract a larger roi, roi_padded. 
     (2) crop a random roi out from the roi_padded.'''
  if random_shift_delta != 0:   # When need to SHIFT   
    roi_param_padded = {}
    roi_param_padded['roi_offset_x'] = roi_param['roi_offset_x'] - random_shift_delta
    roi_param_padded['roi_offset_y'] = roi_param['roi_offset_y'] - random_shift_delta
    roi_param_padded['roi_width'] = roi_param['roi_width'] + 2*random_shift_delta
    roi_param_padded['roi_height'] = roi_param['roi_height'] + 2*random_shift_delta
    roi_padded = extract_roi(image, roi_param_padded)
    # Randomly crop a fixe-sized ROI within the "roi_padded"
    random_crop_size = [roi_param['roi_height'], roi_param['roi_width'], 3]
    roi = tf.image.random_crop(roi_padded, random_crop_size)
  else:  # When NO need to SHIFT (the 'if' portion also handles the else case, but it requires a few extra steps)
    roi = extract_roi(image, roi_param)
  return roi

def zoom_central(image, minimum_central_fraction):

  central_fraction = random.uniform(minimum_central_fraction,1)

  if (minimum_central_fraction!=1 and minimum_central_fraction>0 and minimum_central_fraction<1):
    return tf.image.central_crop(image, central_fraction)
  return image

def extract_roi(image, roi_param):
  """
  This op cuts a rectangular part out of image. The top-left corner of the returned image is at offset_height, offset_width in image, and its lower-right corner is at offset_height + target_height, offset_width + target_width
  Input: 
     image: 4-D Tensor of shape [batch, height, width, channels] 
         or 3-D Tensor of shape [height, width, channels]"""

  # Handel the case when cropping is not needed --> skip cropping  
  if roi_param['roi_height'] == -1 and roi_param['roi_width'] == -1:
    print('No need to extract ROI, when roi_height and roi_width are -1!')
    return image  

  roi = tf.image.crop_to_bounding_box(
    image,
    roi_param['roi_offset_y'],
    roi_param['roi_offset_x'],
    roi_param['roi_height'],
    roi_param['roi_width']
  )
  return roi

def augment_random_rotate(images, random_rotate):
  # print('IMAGES RANK +++++++++++++', images.get_shape().ndims)
  if random_rotate != 0:
    # add rotate transformation to augment images
    rotate_degree = tf.random_uniform(tensor_shape.scalar(), minval=-random_rotate, maxval=random_rotate)
    rotate_radians = rotate_degree * pi / 180
    # THE FOLLOWING FUNCTION REQUIRES THE RANK OF THE INPUT "images" TO BE KNOWN. 
    images = tf.contrib.image.rotate(images, rotate_radians, interpolation='BILINEAR') 

  # if random_rotate:
  #   k = np.random.random_integers(0,3)
  #   images = tf.image.rot90(images, k)
  return images

def augment_flip(image, flip_left_right, flip_up_down):
  if flip_left_right:
    image = tf.image.random_flip_left_right(image)
  if flip_up_down:
    image = tf.image.random_flip_up_down(image)
  return image

def augment_brightness(image, random_brightness):
  # Adjust brightness using a delta randomly picked in the interval [-random_brightness, random_brightness)
  # random_brightness should be in the range [0,1)
  if random_brightness != 0:   
    image = tf.image.random_brightness(image, random_brightness)
  return image 

def input_image_resize(images, input_height, input_width):
  # input image resize to the network's expected input size
  # input image: 4-D with shape [batch, height, width, channels] 
  resized = tf.image.resize_bilinear(images, [input_height, input_width])
  return resized

def input_image_normalization(augmented_image, input_mean, input_std):
  # input normalization 
  offset_image = tf.subtract(augmented_image, input_mean)
  normalized = tf.multiply(offset_image, 1.0 / input_std)
  # normalized = tf.expand_dims(normalized, 0, name='DistortResult')
  return normalized

# @profile
def save_tensor_as_image(tensor, output_file_name):
  from PIL import Image
  image = Image.fromarray(tensor[0].astype(np.uint8))
  # image.save('name.png', format='PNG')
  image.save(output_file_name, format='PNG')

  ### The tensorflow solution below has memory leakage!!!!!!! 
  # tensor_new = tf.cast(tensor, tf.uint8)
  # tensor_new = tf.squeeze(tensor_new) 
  # tensor_new = tf.image.encode_jpeg(tensor_new, quality=100, format='rgb')
  # writer = tf.write_file(output_file_name, tensor_new)
  # sess.run(writer)
  # del writer
  # del tensor_new

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor, nums_samples):
  """Adds a new softmax and fully-connected layer for training.

  We need to retrain the top layer to identify our new classes, so this function
  adds the right operations to the graph, along with some variables to hold the
  weights, and then sets up all the gradients for the backward pass.

  The set up for the softmax and fully-connected layers is based on:
  https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html

  Args:
    class_count: Integer of how many categories of things we're trying to
    recognize.
    final_tensor_name: Name string for the new final node that produces results.
    bottleneck_tensor: The output of the main CNN graph.
    # bottleneck_tensor_size: How many entries in the bottleneck vector.

  Returns:
    The tensors for the training and cross entropy results, and tensors for the
    bottleneck input and ground truth input.
  """
  # bottleneck_tensor_size = get_bottleneck_tensor_size(bottleneck_tensor)
  bottleneck_tensor_size = int(bottleneck_tensor.shape[-1]) # auto detect the size of the bottleneck_tensor
  # print('------------------- bottleneck_tensor_size in add_final_training_ops:', bottleneck_tensor_size)

  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  # def add_dense(tensor, class_count, units=None,**kwargs):
  #   assert class_count > 1
  #   return tf.layers.dense(tensor, units=max(units,class_count),**kwargs)
  def add_dense_layers(tensor, class_count, num_dense_layers):
    # loop over each dense layer that needs to be added 
    for layer_ind in range(1, FLAGS.add_dense_layers+1):
      # Define activation
      if layer_ind != num_dense_layers:
        activation_func = tf.nn.relu 
      elif layer_ind == num_dense_layers:  # last dense layer 
        activation_func = None #tf.nn.softmax

      # Define number of output units in dense layer
      if layer_ind == 1:  # first layer
        size = bottleneck_tensor_size # arbitrarily choose a dimension in range [class_count, bottleneck_tensor_size]
      elif layer_ind != num_dense_layers: # middle layers, reduce dense layer size 
        size = int(size*FLAGS.dense_layers_thinning_factor/100)
      elif layer_ind == num_dense_layers: 
        size = class_count

      layer_name = 'dense_layer_' + str(layer_ind) 
      tensor = tf.layers.dense(tensor, units=size, activation=activation_func, name=layer_name)
      tf.summary.histogram('dense_layer_%d' % (layer_ind), tensor)
    return tensor 

  # Organizing the following ops as `final_training_ops` so they're easier
  # to see in TensorBoard
  with tf.name_scope('final_training_ops'):

    logits = add_dense_layers(bottleneck_input, class_count, FLAGS.add_dense_layers)

    # if FLAGS.feature_vector == 'layer7' or FLAGS.feature_vector == 'L7':
    #   # Di - try adding another fully connected layer, for feature vector = layer7 & mobilenetV1
    #   dense_layer_dim = 512 # arbitrarily chose a dimension
    #   with tf.name_scope('weights'):
    #     initial_value_2 = tf.truncated_normal(
    #         [bottleneck_tensor_size, dense_layer_dim], stddev=0.001)
    #     layer_weights_2 = tf.Variable(initial_value_2, name='final_weights_2')
    #     variable_summaries(layer_weights_2)
    #   with tf.name_scope('biases'):
    #     layer_biases_2 = tf.Variable(tf.zeros([dense_layer_dim]), name='final_biases_2')
    #     variable_summaries(layer_biases_2)
    #   with tf.name_scope('Wx_plus_b'):
    #     # dense_layer_output = tf.matmul(bottleneck_input, layer_weights_2) + layer_biases_2
    #     dense_layer_output = tf.add(tf.matmul(bottleneck_input, layer_weights_2), layer_biases_2) # attempted to fixed ncsdk_v2.08 compilation error
    #     tf.summary.histogram('pre_activations', dense_layer_output)

    #   with tf.name_scope('weights'):
    #     initial_value = tf.truncated_normal(
    #         [dense_layer_dim, class_count], stddev=0.001)
    #     layer_weights = tf.Variable(initial_value, name='final_weights')
    #     variable_summaries(layer_weights)
    #   with tf.name_scope('biases'):
    #     layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
    #     variable_summaries(layer_biases)
    #   with tf.name_scope('Wx_plus_b'):
    #     # logits = tf.matmul(dense_layer_output, layer_weights) + layer_biases
    #     logits = tf.add(tf.matmul(dense_layer_output, layer_weights), layer_biases)
    #     tf.summary.histogram('pre_activations', logits)


  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  def class_weights(nums_samples):
    # Input: a list of integers that represents the number of samples in each class. 
    # Output: a dictionary of weights, keys are numeric values
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    total = sum(nums_samples)
    class_count = len(nums_samples)
    class_weights = [0]*class_count # initialization 
    for i in range(class_count):
      class_weights[i] = (1 / nums_samples[i])*(total)/float(class_count) #(1 / num_samples[i])*(total)/2.0 

    # class_weight = {0: weight_for_pos, 1: weight_for_neg}
    for i in range(class_count):
      print('Weight for class {:2d} : {:.2f}'.format(i, class_weights[i]))
  #  print('Weight for class 1 (good): {:.2f}'.format(class_weights[1]))
    return class_weights 
  
  class_weights = class_weights(nums_samples)
  # cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
  #   labels=ground_truth_input, logits=logits, pos_weight=weight_for_pos, name=None
  # )


  with tf.name_scope('cross_entropy1'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)

    # Add weights for different classes for imbalanced dataset
    weighted_samples = tf.reduce_sum(class_weights * ground_truth_input, axis=1)
    weighted_cross_entropy= cross_entropy * weighted_samples
    with tf.name_scope('total'):
      # cross_entropy_mean = tf.reduce_mean(cross_entropy)
      cross_entropy_mean = tf.reduce_mean(weighted_cross_entropy) 

    # # Diane: add L2 Regularization to loss function
    # beta = 0.01 # 1 #0.1 #0.01
    # regularizer = tf.nn.l2_loss(layer_weights)
    # cross_entropy_mean = tf.reduce_mean(cross_entropy_mean + beta * regularizer) 

  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    if FLAGS.optimizer == 'GD': 
      optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer =='Momentum':
      optimizer = tf.compat.v1.train.MomentumOptimizer(
        FLAGS.learning_rate, momentum=0.9, use_locking=False, name='Momentum', use_nesterov=False
      )
    elif FLAGS.optimizer == 'RMSProp':
      optimizer = tf.compat.v1.train.RMSPropOptimizer(
        FLAGS.learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False,
        centered=False, name='RMSProp'
      )
    elif FLAGS.optimizer == 'Adam':
      optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=FLAGS.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
        name='Adam'
      )
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor, metric='accuracy'):
  """Inserts the operations we need to evaluate the accuracy of our results.

  Args:
    result_tensor: The new final node that produces results.
    ground_truth_tensor: The node we feed ground truth data
    into.

  Returns:
    Tuple of (evaluation step, prediction).
  """
  print('Evaluation metric: ', metric)
  if metric == 'accuracy':
    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        prediction = tf.argmax(result_tensor, 1)
        correct_prediction = tf.equal(
            prediction, tf.argmax(ground_truth_tensor, 1))
      with tf.name_scope('accuracy'):
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    #return evaluation_step, prediction
  elif metric == 'recall':
    with tf.name_scope('recall'):
      with tf.name_scope('correct_prediction'):
        prediction = tf.argmax(result_tensor, 1) 
        ground_truth = tf.argmax(ground_truth_tensor, 1)
        # Positive is defective/bad images --> class index is 0/False
        is_true_positive_tensor = tf.logical_and(tf.equal(ground_truth, 0), tf.equal(prediction, 0))
        is_false_negative_tensor = tf.logical_and(tf.equal(ground_truth, 0), tf.equal(prediction, 1))
       
        true_positive = tf.reduce_sum(tf.cast(is_true_positive_tensor, tf.float32))
        false_negative = tf.reduce_sum(tf.cast(is_false_negative_tensor, tf.float32))
       # evaluation_step, prediction = tf.metrics.recall(labels=ground_truth_tensor, predictions=result_tensor)
      with tf.name_scope('recall'):
        evaluation_step = tf.div(true_positive, (true_positive + false_negative))
    tf.summary.scalar('recall', evaluation_step)
  elif metric == 'precision':
    with tf.name_scope('precision'):
      with tf.name_scope('correct_prediction'):
        prediction = tf.argmax(result_tensor, 1) 
        ground_truth = tf.argmax(ground_truth_tensor, 1)
        # Positive is defective/bad images --> class index is 0
        is_true_positive_tensor = tf.logical_and(tf.equal(ground_truth, 0), tf.equal(prediction, 0))
        is_false_positive_tensor = tf.logical_and(tf.equal(ground_truth, 1), tf.equal(prediction, 0))
       
        true_positive = tf.reduce_sum(tf.cast(is_true_positive_tensor, tf.float32))
        false_positive = tf.reduce_sum(tf.cast(is_false_positive_tensor, tf.float32))
       # evaluation_step, prediction = tf.metrics.recall(labels=ground_truth_tensor, predictions=result_tensor)
      with tf.name_scope('precision'):
        evaluation_step = tf.div(true_positive, (true_positive + false_positive))
    tf.summary.scalar('precision', evaluation_step)
        
  return evaluation_step, prediction

  # acc, acc_op = tf.metrics.accuracy(labels=ground_truth_tensor, predictions=result_tensor)
  # rec, rec_op = tf.metrics.recall(labels=ground_truth_tensor, predictions=result_tensor)
  # pre, pre_op = tf.metrics.precision(labels=ground_truth_tensor, predictions=result_tensor)

  # v = sess.run(acc_op, feed_dict={x: testing_set["target"],y: scoreArr}) #accuracy
  # r = sess.run(rec_op, feed_dict={x: testing_set["target"],y: scoreArr}) #recall
  # p = sess.run(pre_op, feed_dict={x: testing_set["target"],y: scoreArr}) #precision

  # print("accuracy %f", v)
  # print("recall %f", r)
  # print("precision %f", p)
  # accuracy_score = DNNClassifier.evaluate(input_fn=lambda:input_fn(testing_set),steps=1)["accuracy"]

def save_graph_to_file(sess, graph, graph_file_name):
  dir_list = graph_file_name.split('/')[:-1]

  dir_name = '/'.join(dir_list)
  Path(dir_name).mkdir(parents=True, exist_ok=True)
  output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants( #graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with tf.gfile.GFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def prepare_file_system():
  # Setup the directory we'll write summaries to for TensorBoard
  # if tf.gfile.Exists(FLAGS.summaries_dir):
  #   tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)
  if FLAGS.intermediate_store_frequency > 0:
    ensure_dir_exists(FLAGS.intermediate_output_graphs_dir)
  return


def create_model_info(architecture):
  """Given the name of a model architecture, returns information about it.

  There are different base image recognition pretrained models that can be
  retrained using transfer learning, and this function translates from the name
  of a model to the attributes that are needed to download and train with it.

  Args:
    architecture: Name of a model architecture.

  Returns:
    Dictionary of information about the model, or None if the name isn't
    recognized

  Raises:
    ValueError: If architecture name is unknown.
  """
  architecture = architecture.lower()
  if architecture == 'inception_v3':
    # pylint: disable=line-too-long
    data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    bottleneck_tensor_name = 'pool_3/_reshape:0'
    # bottleneck_tensor_size = 2048
    input_width = 299
    input_height = 299
    input_depth = 3
    resized_input_tensor_name = 'Mul:0'
    model_file_name = 'classify_image_graph_def.pb'
    input_mean = 128
    input_std = 128
  elif architecture.startswith('mobilenet_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
        version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
          """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
  but found '%s' for architecture '%s'""",
          version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
        size_string != '160' and size_string != '128'):
      tf.logging.error(
          """The Mobilenet input size should be '224', '192', '160', or '128',
 but found '%s' for architecture '%s'""",
          size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
            "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
            architecture)
        return None
      is_quantized = True
    data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
    data_url += version_string + '_' + size_string + '_frozen.tgz'

    # get bottleneck_tensor_name
    bottleneck_tensor_name = 'MobilenetV1/Logits/AvgPool_1a/AvgPool:0'
    # bottleneck_tensor_name = constant.MOBILENETV1_LAYERS[FLAGS.feature_vector]


    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    resized_input_tensor_name = 'input:0'
    if is_quantized:
      model_base_name = 'quantized_graph.pb'
    else:
      model_base_name = 'frozen_graph.pb'
    model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
    model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  elif architecture.startswith('mobilenetv2_'):
    parts = architecture.split('_')
    if len(parts) != 3 and len(parts) != 4:
      tf.logging.error("Couldn't understand architecture name '%s'",
                       architecture)
      return None
    version_string = parts[1]
    if (version_string != '1.0' and version_string != '0.75' and
            version_string != '0.50' and version_string != '0.25'):
      tf.logging.error(
        """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
but found '%s' for architecture '%s'""",
        version_string, architecture)
      return None
    size_string = parts[2]
    if (size_string != '224' and size_string != '192' and
            size_string != '160' and size_string != '128'):
      tf.logging.error(
        """The Mobilenet V2 input size should be '224', '192', '160', or '128',
but found '%s' for architecture '%s'""",
        size_string, architecture)
      return None
    if len(parts) == 3:
      is_quantized = False
    else:
      if parts[3] != 'quantized':
        tf.logging.error(
          "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
          architecture)
        return None
      is_quantized = True
    data_url = 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_'
    data_url += version_string + '_' + size_string + '.tgz'
    if FLAGS.feature_vector == 'layer6' or FLAGS.feature_vector == 'L6':
      bottleneck_tensor_name = 'MobilenetV2/expanded_conv_6/project/BatchNorm/FusedBatchNorm:0'
      # bottleneck_tensor_size = int(64 * float(version_string))  # 64
    elif FLAGS.feature_vector == 'second_last' or FLAGS.feature_vector == 'L-2':
      bottleneck_tensor_name = 'MobilenetV2/Logits/AvgPool:0'
      # bottleneck_tensor_size = int(1280*float(version_string)) # 1280
    elif FLAGS.feature_vector == 'classification' or FLAGS.feature_vector == 'L-1':
      bottleneck_tensor_name = 'MobilenetV2/Predictions/Reshape:0'
      # bottleneck_tensor_size = 1001
    input_width = int(size_string)
    input_height = int(size_string)
    input_depth = 3
    resized_input_tensor_name = 'input:0'
    if is_quantized:
      model_base_name = 'quantized_graph.pb'
    else:
      model_file_name = 'mobilenet_v2_' + version_string + '_' + size_string + '_frozen.pb' # 'frozen_graph.pb'
    # model_dir_name = 'mobilenet_v2_' + version_string + '_' + size_string
    # model_file_name = os.path.join(model_dir_name, model_base_name)
    input_mean = 127.5
    input_std = 127.5
  else:
    tf.logging.error("Couldn't understand architecture name '%s'", architecture)
    raise ValueError('Unknown architecture', architecture)

  return {
      'data_url': data_url,
      'bottleneck_tensor_name': bottleneck_tensor_name,
      # 'bottleneck_tensor_size': bottleneck_tensor_size,
      'input_width': input_width,
      'input_height': input_height,
      'input_depth': input_depth,
      'resized_input_tensor_name': resized_input_tensor_name,
      'model_file_name': model_file_name,
      'input_mean': input_mean,
      'input_std': input_std,
  }


def add_png_decoding(input_width, input_height, input_depth, input_mean,
                      input_std, roi_param):
  """Adds operations that perform PNG decoding and resizing to the graph..
  The differences from add_jpeg_decoding is that we used decode_png instead of decode_jpeg

  Args:
    input_width: Desired width of the image fed into the recognizer graph.
    input_height: Desired width of the image fed into the recognizer graph.
    input_depth: Desired channels of the image fed into the recognizer graph.
    input_mean: Pixel value that should be zero in the image for the graph.
    input_std: How much to divide the pixel values by before recognition.

  Returns:
    Tensors for the node to feed JPEG data into, and the output of the
      preprocessing steps.
  """
  png_data = tf.placeholder(tf.string, name='DecodePNGInput')
  decoded_image = tf.image.decode_image(png_data, channels=input_depth) # changed to use the more generic decode_image function. # tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  # Diane added extract roi -- no need other augmentations for testing images
  decoded_image_as_float = extract_roi(decoded_image_as_float, roi_param)

  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)

  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)  
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)

  return png_data, mul_image

def deleteOldModels(dir_path):
  max_to_keep = 1  # Maximum number of checkpoints to keep.  As new checkpoints are created, old ones are deleted.

  folder = dir_path
  file_name_patten = 'best_retrained_graph_step*.pb' #'best_retrained_graph_step*.meta'

  files_path = os.path.join(folder, file_name_patten)
  files = sorted( glob.iglob(files_path), key=os.path.getmtime, reverse=True ) # sort files by modified date

  for i in range(max_to_keep, len(files)):  # max_to_keep*3 since there are 3 files per checkpoint
    # path_w_filename = files[i].split(".")[0]
    # files_w_other_ext1 = path_w_filename + '.index'
    # files_w_other_ext2 = path_w_filename + '.data-00000-of-00001'
    # files_w_other_ext3 = path_w_filename + '.pb'

    # print('To delete file: ' + files[i])
    os.remove(files[i])

  return

def get_nums_samples(image_lists):
  # Get number_of_images per class
  # print(image_lists.keys())
  nums_samples = []
  for label_name, label_lists in image_lists.items():
    label_lists = image_lists[label_name]
    nums_samples.append(len(label_lists['training']))
    print('Number of training samples in class "', label_name, '": ', len(label_lists['training'])) # 'training', validation', 'testing'
  # print(nums_samples)
  return nums_samples

def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare necessary directories that can be used during training
  prepare_file_system()

  # Gather information about the model architecture we'll be using.
  model_info = create_model_info(FLAGS.architecture)
  if not model_info:
    tf.logging.error('Did not recognize architecture flag')
    return -1

  # Set up the pre-trained graph.
  maybe_download_and_extract(model_info['data_url'])
  graph, bottleneck_tensor, resized_image_tensor = (
      create_model_graph(model_info))

  # Look at the folder structure, and create lists of training, validation, and testing images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
    return -1
  if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' +
                     FLAGS.image_dir +
                     ' - multiple classes are needed for classification.')
    return -1

  # Remove existing cached bottleneck files
  if FLAGS.update_bottleneck_file == True:  
    for label_name, label_lists in image_lists.items():
      label_lists = image_lists[label_name]
      sub_dir = label_lists['dir']
      sub_dir_path = os.path.join(FLAGS.bottleneck_dir, sub_dir)
      print('bottleneck path to remove:', sub_dir_path)
      if os.path.exists(sub_dir_path):
        shutil.rmtree(sub_dir_path)

  roi_param = {
    'roi_offset_x':FLAGS.roi_offset_x,
    'roi_offset_y':FLAGS.roi_offset_y,
    'roi_width':FLAGS.roi_width,
    'roi_height':FLAGS.roi_height
  }
  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.flip_up_down, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness, FLAGS.random_rotate, FLAGS.random_shift_delta, roi_param)

  # # Di tries to restore and resume training
  # saver = tf.train.Saver()
  # saver.restore(sess, tf.train.latest_checkpoint('./'))  # search for checkpoint file
  # graph = tf.get_default_graph()

  with tf.Session(graph=graph) as sess:
    # Set up the image decoding sub-graph. Use for (1) training images without augmentation & (2) test images.
    jpeg_data_tensor, decoded_image_tensor = add_png_decoding(
        model_info['input_width'], model_info['input_height'],
        model_info['input_depth'], model_info['input_mean'],
        model_info['input_std'], roi_param)  
    if do_distort_images:
      # We will be applying distortions, so setup the operations we'll need.
      (distorted_jpeg_data_tensor, augmented_image_roi_tensor,
       distorted_image_tensor) = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.flip_up_down, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness, FLAGS.random_rotate, model_info['input_width'],
           model_info['input_height'], model_info['input_depth'],
           model_info['input_mean'], model_info['input_std'], roi_param, FLAGS.random_shift_delta, FLAGS.zoom_in)
    else:
      # We'll make sure we've calculated the 'bottleneck' image summaries and
      # cached them on disk.
      cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                        FLAGS.bottleneck_dir, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, FLAGS.architecture)

    # Add the new layer that we'll be training.
    nums_samples = get_nums_samples(image_lists)
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(image_lists.keys()), FLAGS.final_tensor_name, bottleneck_tensor, nums_samples) #model_info['bottleneck_tensor_size']

    # Create the operations we need to evaluate the accuracy of our new layer.
    metric = FLAGS.metric  # 'accuracy', 'recall', precision'
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input, metric)
 

    # for saving .meta models  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model_name = 'tf_files/' + str(FLAGS.dataset_name) + '_' + str(FLAGS.architecture) + '_' + str(FLAGS.feature_vector) \
                 + '_lr' + str(FLAGS.learning_rate) + '_trnBt' + str(FLAGS.train_batch_size) + '_valBt' + \
                 str(FLAGS.validation_batch_size) + '_file_'+ str(FLAGS.append_filename) + '_bestC' + \
                 str(FLAGS.best_result_count_thresh)

    # Merge all the summaries and write them out to the summaries_dir !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/' + model_name[9:] + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir + '/' + model_name[9:] + '/validation')
    print('Model Name: ', model_name[9:])


    # Set up all our weights to their initial default values.
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    accuracy_best = -0.01  # initialization, set to a small negative value since the first computed accuracy could be 0

    # prepare test data - for reuse in the for loop
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, FLAGS.test_batch_size, 'testing',
            FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
            FLAGS.architecture))
    # Run the training for as many cycles as requested on the command line.
    for i in range(FLAGS.epochs):
      # Get a batch of input bottleneck values, either calculated fresh every
      # time with distortions applied, or from the cache stored on disk.
      if do_distort_images:
        # print('Distort/augment image in iteration  ', i, '  ################################')
        (train_bottlenecks,
         train_ground_truth) = get_random_distorted_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.image_dir, distorted_jpeg_data_tensor, augmented_image_roi_tensor, 
             distorted_image_tensor, resized_image_tensor, bottleneck_tensor, i)
      else:
        (train_bottlenecks,
         train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, FLAGS.train_batch_size, 'training',
             FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
             FLAGS.architecture)
      # Feed the bottlenecks and ground truth into the graph, and run a training
      # step. Capture training summaries for TensorBoard with the `merged` op.
      train_summary, _ = sess.run(
          [merged, train_step],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      # Every so often, print out how well the graph is training.
      is_last_step = (i + 1 == FLAGS.epochs)
      if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: Step %d: Train %s = %.2f%%' %
                        (datetime.now(), i, metric, train_accuracy * 100))
        tf.logging.info('%s: Step %d: Cross entropy = %f' %
                        (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.validation_batch_size, 'validation',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
                FLAGS.architecture))
        # Run a validation step and capture training summaries for TensorBoard
        # with the `merged` op.
        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: Step %d: Validation %s = %.2f%% (N=%d)' %
                        (datetime.now(), i, metric, validation_accuracy * 100,
                         len(validation_bottlenecks)))
        # Store/update best results, based on validation accuracy. Sometimes, validation accuracy in some very early iteration is high, while training accuracy is low. Perhaps, using combined validation accuracy & training accuracy is another good measurement
        accuracy_current = validation_accuracy # (validation_accuracy + train_accuracy)/2  #validation_accuracy is used as the measurement
        if ( (accuracy_current >= accuracy_best-0.015) and (train_accuracy >= validation_accuracy * 0.9) ): #!!!! COMPUTE TEST ACCURACY MORE OFTEN
          # check test accuracy !!!! (Just for observation, not for making decisions!!!!)
          # test_accuracy, predictions = sess.run(
          #     [evaluation_step, prediction],
          #     feed_dict={bottleneck_input: test_bottlenecks,
          #                ground_truth_input: test_ground_truth})
          # print('##################### Test Accuracy %s = %.2f%% (N=%d)' % (metric, test_accuracy * 100, len(test_bottlenecks)))

          # save model
          # model_name_best = 'tf_files/best_retrained_graph_step' + str(i)
          # # save model - checkpoint
          # model_save_path = saver.save(sess, model_name_best)

          # save model - frozen graph
          # tf.logging.info('Save best result to : ' + FLAGS.best_output_graph)
          # save_graph_to_file(sess, graph, FLAGS.best_output_graph)
          model_name_best = os.path.join(FLAGS.output_graph, 'best_retrained_graph_step' + str(i) + '.pb')
          # model_name_best = 'tf_files/best_retrained_graph_step' + str(i) + '.pb'
          save_graph_to_file(sess, graph, model_name_best)
          tf.logging.info('Save model to : ' + model_name_best)


          deleteOldModels(FLAGS.output_graph)   # keep the latest "max_to_keep" models, and delete old ones

          # update best result counter
          if (accuracy_current > accuracy_best): # reset best_result_count
            best_result_count = 1
          else:
            best_result_count = best_result_count + 1
            print('best_result_count:  ', best_result_count)
          accuracy_best = accuracy_current

          # reduce learning rate
          if (best_result_count % BEST_RESULT_COUNT_FOR_LEARNING_RATE_REDUCTION == 0 and best_result_count > 0 ):  #FLAGS.best_result_count_thresh_reduce_learning_rate):
            FLAGS.learning_rate = FLAGS.learning_rate * LEARNING_RATE_REDUCTION_RATIO
            print('Reduce learning rate! New learning rate is: %.3e' % FLAGS.learning_rate )
          # # save best result, check stop condition
          # if (best_result_count >= FLAGS.best_result_count_thresh):
          #   print('Reached best result count threshhold! Stop training, and save the model!')
          #   break

      # Store intermediate results
      intermediate_frequency = FLAGS.intermediate_store_frequency

      if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
          and i > 0):
        intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                  'intermediate_' + str(i) + '.pb')
        tf.logging.info('Save intermediate result to : ' +
                        intermediate_file_name)
        save_graph_to_file(sess, graph, intermediate_file_name)


    # We've completed all our training, so run a final test evaluation on
    # some new images we haven't used before.
    # THIS HAS ALREADY BEEN DEFINED EARLIER
    # test_bottlenecks, test_ground_truth, test_filenames = (
    #     get_random_cached_bottlenecks(
    #         sess, image_lists, FLAGS.test_batch_size, 'testing',
    #         FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
    #         decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
    #         FLAGS.architecture))


    if FLAGS.print_misclassified_test_images:
      test_accuracy, predictions = sess.run(
          [evaluation_step, prediction],
          feed_dict={bottleneck_input: test_bottlenecks,
                     ground_truth_input: test_ground_truth})
      tf.logging.info('Final test %s = %.2f%% (N=%d)' %
                      (metric, test_accuracy * 100, len(test_bottlenecks)))

      tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
      for i, test_filename in enumerate(test_filenames):
        if predictions[i] != test_ground_truth[i].argmax():
          tf.logging.info('%70s  %s' %
                          (test_filename,
                           list(image_lists.keys())[predictions[i]]))


    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]  # THIS HELPS IDENTIFY NODE
    # save graph def file
    
    
    # print('model name:  ', model_name)
    # save model in meta format
    # saver = tf.train.Saver()
    # model_save_path = saver.save(sess, model_name)

    # save model in pb format
    # save_graph_to_file(sess, graph, model_name + '.pb')

    # Write out the trained graph and labels with the weights stored as constants.
    # output_graph_file = os.path.join(FLAGS.output_graph,'ouput_trained_model.pb')

    # save_graph_to_file(sess, graph, output_graph_file)
    with tf.gfile.GFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()


  parser.add_argument(
      '--dataset_name',
      type=str,
      default="blocks",
      help='Short name to distingish datasets'
  )

  parser.add_argument(
      '--best_output_graph',
      type=str,
      default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR'), 'tf_files/best_retrained_graph.pb'),
      help='Where to save the best graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR'), 'tf_files/intermediate_graph/'),
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR'), 'tf_files/labels.txt'),
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default=os.path.join(os.environ.get('SM_OUTPUT_DATA_DIR'), 'tf_files/training_summaries'),
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=1000,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--best_result_count_thresh',
      type=int,
      default=100,
      help='How many best results reached before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.001,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=0,
      help='What percentage of images to use as a test set.'
  )
  parser.add_argument(
      '--validation_percentage',
      type=int,
      default=20,
      help='What percentage of images to use as a validation set.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=128,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--validation_batch_size',
      type=int,
      default=-1, #100,
      help="""\
      How many images to use in an evaluation batch. This validation set is
      used much more often than the test set, and is an early indicator of how
      accurate the model is during training.
      A value of -1 causes the entire validation set to be used, which leads to
      more stable results across training iterations, but may be slower on large
      training sets.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--pretrain_model_dir',
      type=str,
      default='model',  #'/tmp/imagenet',
      help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='tf_files/bottlenecks',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=False,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--flip_up_down',
      default=False,
      help="""\
      Whether to randomly flip half of the training images vertically.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=10,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=10,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=float,
      default=10,
      help="""\
      A value delta determining how much to randomly multiply the training image
      input pixels up or down by. The value should be in the range [0,1)\
      """
  )
  parser.add_argument(
      '--random_rotate',
      type=float,
      default=10,
      help="""\
      A degree determining how much to randomly rotate the training image
      clockwise or counter-clockwise by.\
      """
  )
  parser.add_argument(
      '--architecture',
      type=str,
      default='mobilenet_1.0_224',
      help="""\
      Which model architecture to use. 'inception_v3' is the most accurate, but
      also the slowest. For faster or smaller models, chose a MobileNet with the
      form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
      'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
      pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
      less accurate, but smaller and faster network that's 920 KB on disk and
      takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
      for more information on Mobilenet.\
      """)
  parser.add_argument(
      '--append_filename',
      type=str,
      default='',
      help="""\
      Append a string on file names in order to shuffle files in hash process when partition into training, validation,
      and testing data sets.\
      """
  )
  parser.add_argument(
      '--feature_vector',
      type=str,
      default='L-2', # ('second_last' or "L-2"),  # 'layer7' (or "L7"), 'classification' (or "L-1")-- with prediction on 1001 classes
      # choices=constant.MOBILENETV1_LAYERS.keys(),
      help="""\
      Define transfer learn topology. Either remove or keep the last layer.\
      """
  )
  parser.add_argument(
      '--update_bottleneck_file',
      default=False,  # False  # True
      help="""\
      If set True, remove previous cached bottleneck files, and create new ones. This is needed when bottleneck layer is redefined to have different size or value.\
      """
  )
  parser.add_argument(
      '--proper_data_partition',
      default=True,  #
      help="""\
      If set True, all augmented data from the same input are included in only one of the train, validation, or test set.\
      """
  )
  parser.add_argument(
      '--hash_full_path',
      default=False,  # True
      help="""\
      If set True, hash full path name during data partition; if set False, hash base name.\
      """
  )
  parser.add_argument(
      '--roi_offset_x', 
      type=int, 
      default=0, 
      help='x_min of ROI, zero-based'
  )
  parser.add_argument(
      '--roi_offset_y', 
      type=int, 
      default=0, 
      help='y_min of ROI, zero-based'
  )
  parser.add_argument(
      '--roi_width', 
      type=int, 
      default=-1, 
      help='width of ROI, one-based'
  )
  parser.add_argument(
      '--roi_height', 
      type=int, 
      default=-1, 
      help='height of ROI, one-based'
  )
  parser.add_argument(
      '--random_shift_delta', 
      type=int, 
      default=0, 
      help='number of pixels random shift is needed'
  )
  parser.add_argument(
      '--zoom_in', 
      type=float, 
      default=1.0, 
      help='fraction of size to crop Usage'
  )
  parser.add_argument(
      '--visualize_augmented_data', 
      default=False, 
      help='Save augmented image as "test_augmented_image_data.jpg" for visualization if this flag is set to be "True"'
  )
  parser.add_argument(
      '--optimizer', 
      default='Adam', 
      help='optimizer can be chosen from "GD", "Momentum", "RMSProp", "Adam" '
  )
  parser.add_argument(
      '--add_dense_layers',
      default=2,
      type=int,
      help="Number of hidden dense layers prior to classifier."
  )
  parser.add_argument(
      '--dense_layers_thinning_factor',
      default=50,
      type=int,
      help="""\
    Percentage of neuron count to keep in progressive dense layers.\
    """
  )
  parser.add_argument(
      '--metric',
      default='accuracy',
      type=str,
      help="""\
    Evaluation metric. Choose from 'accuracy', 'recall', or 'precision'\
    """
  )
  parser.add_argument(
      '--model_dir', 
      type=str, 
      help= 's3 bucket path where the model is stored after training job is complete.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default=os.environ.get('SM_MODEL_DIR'),
      help='local path where the training job writes the model artificats to.'
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default=os.environ.get('SM_CHANNEL_TRAINING'),
      help='local path to directory that contains the input data.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
