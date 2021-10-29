# Import Data from Google Drive 

from google.colab import drive
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import random
import seaborn as sns
import time
import os
import datetime

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

ccdata_df = pd.read_csv(base_dir + 'creditcard.csv', delimiter = ',')
ccdata_df.head()

# Visualization / Plots / Tables / Statistical Summary

### How many Instances of Fraud 

total_fraud_instances = ccdata_df['Class'].sum()
total_fraud_instances

#Get # of total rows
total_rows = ccdata_df['V1'].count()
print(total_rows + 1)
total_rows = total_rows + 1

### Percentage of Fraud to Genuine Transactions

fraud_ratio = total_fraud_instances/total_rows 
print(fraud_ratio)
fraud_percentage = fraud_ratio*100

# fraud_ratio = 0.00017274795 ....
# Instances of fraud are less than 1 percent 


### Histograms for Each Principle Component
