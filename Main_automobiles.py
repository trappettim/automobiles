import pandas as pd
import numpy as np

"""PRINTING SETTINGS FOR DATAFRAMES AND ARRAYS"""
"""these instructions allow to see the all the columns of the dataframe on the run terminal.
To print all the rows use the method: print(main_df.to_string())"""

desired_width = 500
pd.set_option('display.width', desired_width)  # display width
pd.set_option('display.max_columns', 30)  # number of columns
"""np.set_printoptions(linewidth=desired_width)"""  # numpy set display width

"""IMPORTING DATASET INTO PANDAS DATAFRAME"""

url = 'imports-85.data'
headers_names = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
                 "num-of-doors", "body-style", "drive-wheels", "engine-location",
                 "wheel-base", "length", "width", "height", "curb-weight",
                 "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
                 "bore", "stroke", "compression-ratio", "hoursepower", "peak-rpm",
                 "city-mpg", "highway-mpg", "price"]

# reads csv, no headers in the file, puts headers_names as headers, missing values identified by '?':
main_df = pd.read_csv(url, header=None, names=headers_names, na_values='?')

"""print(main_df.columns)"""  # to print the headers
"""print(main_df.to_string())"""  # to print the entire dataframe

# drops rows with missing values in the column 'price'. Modifies the original df
main_df.dropna(subset=['price'], inplace=True)

# finds and prints the rows where the column 'normalized-losses' is NaN
"""print(main_df.loc[main_df['normalized-losses'].isnull()])"""

# replaces the NaN values in the 'normalizes-losses' column with the mean of the column
mean = main_df['normalized-losses'].mean()
main_df['normalized-losses'] = main_df['normalized-losses'].replace(np.nan, mean)

# how to one-hot encode the fuel type column (turning categorical values into numerical ones)
dummy = pd.get_dummies(main_df['fuel-type'])  # creates a dummy object
main_df = pd.concat([main_df, dummy], axis=1)  # concatenates the dummy object with the df

# Binning
bins = np.linspace(main_df['price'].min(),main_df['price'].max(),4) # creates np arrays with 4 equally spaced numbers
group_names = ['Low','Medium','High'] # list with the label names
main_df['price_binned'] = pd.cut(main_df['price'],bins,labels=group_names,include_lowest=True) # creates the new column
print(main_df)