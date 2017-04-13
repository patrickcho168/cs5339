#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Images binary classifier based on scikit-learn SVM classifier.
It uses the RGB color space as feature vector.
'''

from __future__ import division
from __future__ import print_function
from PIL import Image
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import svm
from sklearn import metrics
from sklearn.externals import joblib
from StringIO import StringIO
from urlparse import urlparse
import urllib2
import sys
import os
import pandas as pd
import datetime
from scipy.misc import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

trainFolder = "train/"
fishFolders = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
testFolder = "test_stg2/"

def process_directory(directory):
    '''Returns an array of feature vectors for all the image files in a
    directory (and all its subdirectories). Symbolic links are ignored.

    Args:
      directory (str): directory to process.

    Returns:
      list of list of float: a list of feature vectors.
    '''
    training = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            img_feature = process_image_file(file_path)
            if img_feature != None:
                training.append(img_feature)
    return training


def process_image_file(image_path):
    '''Given an image path it returns its feature vector.

    Args:
      image_path (str): path of the image file to process.

    Returns:
      list of float: feature vector on success, None otherwise.
    '''
    try:
        y = imread(image_path)
        h,w,c = y.shape
        x = resize(y, (16, 16), preserve_range=False)
        # plt.imshow(x)
        # plt.show()
        return x.flatten().flatten()
    except IOError:
        return None
    # image_fp = StringIO(open(image_path, 'rb').read())
    # try:
    #     image = Image.open(image_fp)
    #     return process_image(image)
    # except IOError:
    #     return None


def process_image_url(image_url):
    '''Given an image URL it returns its feature vector

    Args:
      image_url (str): url of the image to process.

    Returns:
      list of float: feature vector.

    Raises:
      Any exception raised by urllib2 requests.

      IOError: if the URL does not point to a valid file.
    '''
    parsed_url = urlparse(image_url)
    request = urllib2.Request(image_url)
    # set a User-Agent and Referer to work around servers that block a typical
    # user agents and hotlinking. Sorry, it's for science!
    request.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux ' \
            'x86_64; rv:31.0) Gecko/20100101 Firefox/31.0')
    request.add_header('Referrer', parsed_url.netloc)
    # Wrap network data in StringIO so that it looks like a file
    net_data = StringIO(urllib2.build_opener().open(request).read())
    image = Image.open(net_data)
    return process_image(image)


def process_image(image, blocks=4):
    '''Given a PIL Image object it returns its feature vector.

    Args:
      image (PIL.Image): image to process.
      blocks (int, optional): number of block to subdivide the RGB space into.

    Returns:
      list of float: feature vector if successful. None if the image is not
      RGB.
    '''
    if not image.mode == 'RGB':
        return None
    print (image.getdata())
    return image.getdata()
    # feature = [0] * blocks * blocks * blocks
    # pixel_count = 0
    # for pixel in image.getdata():
    #     ridx = int(pixel[0]/(256/blocks))
    #     gidx = int(pixel[1]/(256/blocks))
    #     bidx = int(pixel[2]/(256/blocks))
    #     idx = ridx + gidx * blocks + bidx * blocks * blocks
    #     feature[idx] += 1
    #     pixel_count += 1
    # return [x/pixel_count for x in feature]


def show_usage():
    '''Prints how to use this program
    '''
    print("Usage: %s [class A images directory] [class B images directory]" %
            sys.argv[0])
    sys.exit(1)


def train(print_metrics=True):
    '''Trains a classifier. training_path_a and training_path_b should be
    directory paths and each of them should not be a subdirectory of the other
    one. training_path_a and training_path_b are processed by
    process_directory().

    Args:
      training_path_a (str): directory containing sample images of class A.
      training_path_b (str): directory containing sample images of class B.
      print_metrics  (boolean, optional): if True, print statistics about
        classifier performance.

    Returns:
      A classifier (sklearn.svm.SVC).
    '''
    data = []
    target = []
    idx = 0
    for fishFolder in fishFolders:
        folderName = trainFolder + fishFolder
        if not os.path.isdir(folderName):
            raise IOError('%s is not a directory' % folderName)
        print (folderName)
        training = process_directory(folderName)
        data += training
        target += [idx] * len(training)
        print(idx)
        idx += 1

    # split training data in a train set and a test set. The test set will
    # containt 20% of the total
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,
            target, test_size=0.20) # TODO: CHANGE BACK TO 0.3?
    # define the parameter search space
    parameters = {'kernel': ['linear'], 'C': [1, 10, 100, 1000],
            'gamma': [0.01, 0.001, 0.0001]}
    # search for the best classifier within the search space and return it
    clf = grid_search.GridSearchCV(svm.SVC(probability=True), parameters, scoring='neg_log_loss').fit(x_train, y_train)
    classifier = clf.best_estimator_
    if print_metrics:
        print()
        print('Parameters:', clf.best_params_)
        print()
        print('Best classifier score')
        print(metrics.classification_report(y_test,
            classifier.predict(x_test)))
        print(metrics.log_loss(y_test, classifier.predict_proba(x_test)))
    return classifier

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

def main():
    print('Training classifier...')
    classifier = train()
    
    # Save Classifier
    now = datetime.datetime.now()
    paramsFilename = 'params2' + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.pkl'
    _ = joblib.dump(classifier, paramsFilename, compress=9)

    # # Load Classifier
    # classifier = joblib.load(paramsFilename)

    # Generate Test Results
    
    predictions = []
    testIds = []
    for root, _, files in os.walk(testFolder):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            features = process_image_file(file_path)
            flbase = os.path.basename(file_path)
            testIds.append(flbase)
            prediction = classifier.predict_proba(features)
            
            predictions.append(prediction[0])
    print (predictions)
    print (testIds)
    create_submission(predictions, testIds, "2")
    # while True:
    #     try:
    #         print("Input an image file (enter to exit): "),
    #         file_name = raw_input()
    #         if not file_name:
    #             break
    #         features = process_image_file(file_name)
    #         print(classifier.predict(features))
    #     except (KeyboardInterrupt, EOFError):
    #         break
    #     except:
    #         exception = sys.exc_info()[0]
    #         print(exception)

if __name__ == '__main__':
    main()