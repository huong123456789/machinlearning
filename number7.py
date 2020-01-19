from __future__ import print_function 
import numpy as np 
from sklearn.datasets import fetch_mldata
data_dir = '../../data' # path to your data folder 
mnist = fetch_mldata('MNIST original', data_home=data_dir) 
print("Shape of minst data:", mnist.data.shape)
