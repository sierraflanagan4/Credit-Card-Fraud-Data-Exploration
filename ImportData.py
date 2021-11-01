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
import math

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
print("Total rows of data: " + str(total_rows + 1))
total_rows = total_rows + 1

### Percentage of Fraud to Genuine Transactions

fraud_ratio = total_fraud_instances/total_rows 
print("Fraud ratio: " + str(fraud_ratio))
fraud_percentage = fraud_ratio*100
print("Percentage of fraud in dataset: " + str(fraud_percentage))

# fraud_ratio = 0.00017274795 ....
# Instances of fraud are less than 1 percent 

ccdata_df.hist(column='Amount', grid=True, bins = 40)

ccdata_df.plot('Time', 'Amount')

# Plot histogram of transaction amounts and transactions over time

%%capture

distribution_amount_times, ax = plt.subplots(1,2, figsize=(18,4))

time_val = ccdata_df['Time']

#amount_val = ccdata_df[ccdata_df.Time]['Amount'].sample(n=1000).values

sns.distplot(ccdata_df['Amount'], ax=ax[0], color='r', hist=True, kde=False, bins =30)
ax[0].set_title('Distribution of Transaction Amounts', fontsize = 14)
ax[0].set_xlim([min(ccdata_df['Amount']), max(ccdata_df['Amount'])])
ax[0].set(xlabel = 'Amount', ylabel = "Number of transactions")

#Transform seconds into hours
sns.distplot(time_val/3600, ax=ax[1], color='b', bins=24, hist = True, kde=False)
ax[1].set_title('Distribution of transaction times', fontsize=14)
ax[1].set_xlim([min(time_val/3600), max(time_val/3600)])
#ax[1].set_xticks(range(24))
ax[1].set(xlabel = "Time (hours)", ylabel = "Number of Transactions")

distribution_amount_times


# Add in a column for the hour the transaction belongs to 

ccdata_df['Time_Hour'] = ccdata_df['Time'] / 3600

ccdata_df['Time_Hour'] = ccdata_df['Time_Hour'].apply(np.ceil)

def get_stats(ccdata_df):
  #number of transactions per hour
  nb_tx_per_hour = ccdata_df.groupby(['Time_Hour'])['Time'].count()
  #number of fraudulent transactions per hour
  nb_fraud_per_hour = ccdata_df.groupby(['Time_Hour'])['Class'].sum()

  return(nb_tx_per_hour, nb_fraud_per_hour)

(nb_tx_per_hour, nb_fraud_per_hour) = get_stats(ccdata_df)

n_hours = len(nb_tx_per_hour)
tx_stats = pd.DataFrame({"value":pd.concat([nb_tx_per_hour/50, nb_fraud_per_hour])})
tx_stats['stat_type']=['nb_tx_per_hour']*n_hours+["nb_fraud_per_hour"]*n_hours
tx_stats = tx_stats.reset_index()

#Get statistics for number of fraud transactions per hour

#def get_stats(ccdata_df):
    # Number of transactions per hour


# Plot Fraud relative to Amount and Time
%%capture

sns.set(style='darkgrid')
sns.set(font_scale=1.4)

fraud_and_transactions_statis_fig=plt.gcf()

fraud_and_transactions_statis_fig.set_size_inches(15, 8)

sns_plot = sns.lineplot(x='Time_Hour', y='value', data=tx_stats, hue="stat_type", hue_order=["nb_tx_per_hour", "nb_fraud_per_hour"], legend=False)

sns_plot.set_title('Total Transactions and Number of Fraudulent Transactions Per Hour over 2 Days', fontsize=20)
sns_plot.set(xlabel = 'Number of hours since beginning of data collection', ylabel = 'Number')

labels_legend = ["# Transactions Per Hour (/50)", "# of Fraudulent Transactions Per Hour"]

sns_plot.legend(loc='upper left', labels=labels_legend, bbox_to_anchor=(1.05, 1), fontsize=15)

fraud_and_transactions_statis_fig

### NEXT ### 

### Area under the curve calculations ### 
### may do this in separate script ###  

### Applying Machine Learning Algorithms ### 
### will likely also do this in separate script ###
