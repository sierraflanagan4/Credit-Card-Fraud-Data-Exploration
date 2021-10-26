# Import Data from Google Drive 

from google.colab import drive
import numpy as np
import scipy.io

drive.mount('/content/gdrive', force_remount=True)
root_dr = "/content/gdrive/My Drive/"
base_dir = root_dr + 'DataScience/'
import os
import sys
sys.path.append(base_dir)
print(os.getcwd())
print(os.listdir())
os.chdir(base_dir)
print(os.getcwd())
print(os.listdir())

ccdata = np.genfromtxt(base_dir + 'creditcard.csv', delimiter = ',', skip_header=1)
# URL to Kaggle
# https://www.kaggle.com/mlg-ulb/creditcardfraud
