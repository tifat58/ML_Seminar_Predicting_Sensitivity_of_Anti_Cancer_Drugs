from __future__ import print_function

import pickle
import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tkinter import *
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


tf.reset_default_graph()

saver = tf.train.import_meta_graph('best_model')

y_pred = []

with tf.Session() as sess:

    saver.restore(sess, "best_model.ckpt")
    x = tf.get_collection("x")[0]
    output = sess.run([y_pred], feed_dict={x: x})