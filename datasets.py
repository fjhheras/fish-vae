# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Taken from tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
# Changed by fjhheras@gmail.com for his own fishy purposes 
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

"""Functions for reading fishyfish data."""
import os
import collections
import numpy as np
import cv2

PX_DEPTH = 255.0

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])

def extract_images(folder,expected_size):
    """Extract the images into a 3D numpy array [index, y, x].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 3D float32 numpy array [index, y, x, depth].
    """
    try:
        (width,height) = expected_size
    except:
        print ("WTF in expected_size")
        return -1
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), height,width),
                         dtype=np.float32)
    image_index = 0
    for image in os.listdir(folder):
        image_file = os.path.join(folder, image)
        try:
            #image_data = cv2.equalizeHist(cv2.cvtColor(cv2.imread(image_file),cv2.COLOR_BGR2GRAY))
            image_data = cv2.cvtColor(cv2.imread(image_file),cv2.COLOR_BGR2GRAY)
            image_data = (image_data.astype(np.float32) )/ PX_DEPTH
                    #- PX_DEPTH / 2) / PX_DEPTH
            if image_data.shape != (height,width):
                raise Exception('Unexpected image shape: {} instead of {}'.format(image_data.shape,(height,width)))
            
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read file - it\'s ok, skipping.')
    
    num_images = image_index
    print("Extracting {} images from folder '{}'".format(num_images,folder))
    dataset = dataset[0:num_images, :, :]
    return dataset#.astype(np.float32)

class DataSet(object):

    def __init__(self,
               images,
               reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        if reshape:
            #assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = images.shape[0]
    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            #self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end]


def read_data_sets(train_dir,
                   reshape=True,
                   test_fraction = 0.05,
                   validation_fraction=0.1,
                   expected_size = None):
    
    images = extract_images(train_dir,expected_size)
    num_images = np.shape(images)[0]

    test_images = images[0:np.int32(num_images*test_fraction)]
    train_images = images[np.int32(num_images*test_fraction):-np.int32(num_images*validation_fraction)]
    validation_images = images[-np.int32(num_images*validation_fraction):]
    
    print ("I have {} train, {} validation, {} test images".format(train_images.shape[0], 
        validation_images.shape[0],test_images.shape[0]))


    train = DataSet(train_images, reshape=reshape)
    validation = DataSet(validation_images,
                       reshape=reshape)
    test = DataSet(test_images, reshape=reshape)

    return Datasets(train=train, validation=validation, test=test)


def read_fishyfish():
    return read_data_sets('images/',reshape = True,expected_size = (80,60)) 


