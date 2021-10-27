# Code Modified from Machine Learning For Credit Card Fraud Detection Practical Handbook
# https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


#Function to Generate Customer Profiles

def generate_customer_profiles_table(n_customers, random_state=0):
    
    np.random.seed(random_state)
        
    customer_id_properties=[]
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):
        
        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        
        mean_amount = np.random.uniform(5,100) # Arbitrary (but sensible) value 
        std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
        mean_nb_tx_per_day = np.random.uniform(0,4) # Arbitrary (but sensible) value 
        
        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])
        
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                      'x_customer_id', 'y_customer_id',
                                                                      'mean_amount', 'std_amount',
                                                                      'mean_nb_tx_per_day'])
    
    return customer_profiles_table
  
  
# Generate 5 Example Profiles
n_customers = 5
customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
customer_profiles_table


# Function to Generate Terminal Profiles

def generate_terminal_profiles_table(n_terminals, random_state=0):
    
    np.random.seed(random_state)
        
    terminal_id_properties=[]
    
    # Generate terminal properties from random distributions 
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0,100)
        y_terminal_id = np.random.uniform(0,100)
        
        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])
                                       
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                      'x_terminal_id', 'y_terminal_id'])
    
    return terminal_profiles_table
  
  
# Generate Table for 5 Terminals 

n_terminals = 5
terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 0)
terminal_profiles_table


# Associating Customer and Terminal Profiles
# Function uses Customer location and list of all Terminal Locations to determine which terminals a customer may use

def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals
  
  
# Get list of terminals within radius R = 50 of last customer

# We first get the geographical locations of all terminals as a numpy array
x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
# And get the list of terminals within radius of $50$ for the last customer
get_list_terminals_within_radius(customer_profiles_table.iloc[4], x_y_terminals=x_y_terminals, r=50)


# PLOT: locations of all terminals (red) //  location of the last customer (blue) // region within radius of 50 of the first customer (green)

%%capture

terminals_available_to_customer_fig, ax = plt.subplots(figsize=(5,5))

# Plot locations of terminals
ax.scatter(terminal_profiles_table.x_terminal_id.values, 
           terminal_profiles_table.y_terminal_id.values, 
           color='blue', label = 'Locations of terminals')

# Plot location of the last customer
customer_id=4
ax.scatter(customer_profiles_table.iloc[customer_id].x_customer_id, 
           customer_profiles_table.iloc[customer_id].y_customer_id, 
           color='red',label="Location of last customer")

ax.legend(loc = 'upper left', bbox_to_anchor=(1.05, 1))

# Plot the region within a radius of 50 of the last customr
circ = plt.Circle((customer_profiles_table.iloc[customer_id].x_customer_id,
                   customer_profiles_table.iloc[customer_id].y_customer_id), radius=50, color='g', alpha=0.2)
ax.add_patch(circ)

fontsize=15

ax.set_title("Green circle: \n Terminals within a radius of 50 \n of the last customer")
ax.set_xlim([0, 100])
ax.set_ylim([0, 100])
    
ax.set_xlabel('x_terminal_id', fontsize=fontsize)
ax.set_ylabel('y_terminal_id', fontsize=fontsize)


terminals_available_to_customer_fig


customer_profiles_table['available_terminals']=customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=50), axis=1)
customer_profiles_table

# Create function that will generate transaction data

def generate_transactions_table(customer_profile, start_date ='2018-04-01', nb_days = 10):

    customer_transactions = []

    random.seed(customer_profile.CUSTOMER_ID)
    np.random.seed(customer_profile.CUSTOMER_ID)

    #For all days
    for day in range(nb_days):

      # Random number of transactions for that day
      nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)

      #If nb_tx positive, let us generate transactions
      if nb_tx>0:

        for tx in range(nb_tx):

          # Time of transaction: Around noon, std 20000 seconds. This choice aims at stimulating the fact that
          # most transactions happen during the day.
          time_tx = int(np.random.normal(86400/2, 20000))

          # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
          if (time_tx>0) and (time_tx<86400):

            #Amount is drawn from a normal distribution
            amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)

          # If amount negative, draw from a uniform distribution
            if amount<0:
              amount = np.random.uniform(0, customer_profile.mean_amount*2)

            amount = np.round(amount, decimals=2)

            if len(customer_profile.available_terminals)>0:

              terminal_id = random.choice(customer_profile.avaialable_terminals)

              customer_transactions.append([time_tx+day*86400, day, customer_profile.CUSTOMER_ID,
                                            terminal_id, amount])
              
      customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])

      if len(customer_transactions)>0:
          customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
          customer_transactions=customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]

      return customer_transactions
    
    
    
### TROUBLESHOOTING ###
#type(customer_profiles_table.iloc[0])
#Series.attrs(customer_profiles_table.iloc[0])
#customer_profiles_table.iloc[0].available_terminals


# EX) Generate transactions for the 1st customer, for 5 days, starting at the date 2018-04-01

transaction_table_customer_0=generate_transactions_table(customer_profiles_table.iloc[0],
                                                         start_date = "2018-04-01",
                                                         nb_days = 5)

transaction_table_customer_0

## Gettin an error here saying "'Series' object has no attribute 'avaialable_terminals'"
## I think this is originating from the 'customer_profiles_table.iloc[0]' which is a pandas Series datatype
## Need to resolve error before continuing
