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
                 "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
                 "city-mpg", "highway-mpg", "price"]

# reads csv, no headers in the file, puts headers_names as headers, missing values identified by '?':
main_df = pd.read_csv(url, header=None, names=headers_names, na_values='?')

"""print(main_df.columns)"""  # to print the headers
"""print(main_df.to_string())"""  # to print the entire dataframe
"""print(main_df)""" # to print the first and the last 30 rows

"""FINDING MISSING VALUES"""

#  to replace '?' with NaN if we hadn't done before
"""main_df.replace("?", np.nan, inplace = True)"""

# to print the rows where specified column is NaN
"""print(main_df.loc[main_df['normalized-losses'].isnull()])"""

# to count the missing values in every column
"""missing_values = main_df.isnull()  # new df where every value in the df is a boolean: True if NaN
for column in missing_values.columns.values.tolist():  # loops on the list of columns names. tolist is to convert a-
    print(column)                                      # -pandas series (df columns are series) into a python list
    print (missing_values[column].value_counts())  # counts unique values in the column"""

"""DROP THE WHOLE ROW WHERE MISSING VALUES IN IMPORTANT COLUMNS"""

"""price is the column we want to predict. We can't have missing point here"""

# drops rows with missing values in the column 'price'. Modifies the original df
main_df.dropna(subset=['price'], inplace=True)

"""REPLACE MISSING VALUE WITH MEAN VALUES"""

Col_mean_val = ['normalized-losses','bore', 'stroke','horsepower','peak-rpm']

# replaces the NaN values in all the columns in Col_mean_val with the mean of the column
for column in Col_mean_val:
    mean = main_df[column].astype("float").mean(axis=0)
    main_df[column].replace(np.nan, mean,inplace=True)

"""REPLACE MISSING VALUE WITH FREQUENCY"""

"""first method
mode = main_df['num-of-doors'].mode()  # returns a pandas series with one element, the mode
main_df['num-of-doors'].replace(np.nan, mode[0],inplace=True) # replaces NaN with the only element of the series mode
"""

"""second method"""
mode = main_df['num-of-doors'].value_counts().idxmax() # value_counts for counting unique values. idxmax for mode
main_df['num-of-doors'].replace(np.nan, mode,inplace=True)  # replaces NaN with the only element of the series mode


"""CORRECT DATA FORMAT"""

"""print(main_df.dtypes) # to find out the data type of each column of the df
dft = main_df.dtypes.tolist()  # to store the types before the changes. Later we will print a table with the canges"""

main_df[["bore", "stroke"]] = main_df[["bore", "stroke"]].astype("float")  # to change the type to float
main_df[["normalized-losses"]] = main_df[["normalized-losses"]].astype("int")  
main_df[["price"]] = main_df[["price"]].astype("float")
main_df[["peak-rpm"]] = main_df[["peak-rpm"]].astype("float")

"""test = pd.DataFrame([dft,main_df.dtypes.tolist()],columns=headers_names)  # to create a table with changes
print(test)"""

"""DATA STANDARDIZATION"""

"""all the column must be in the correct unit"""

# transform the data in the column via a mathematical operation
main_df["city-mpg"] = 235 / main_df["city-mpg"]

# rename column
main_df.rename(columns={'city-mpg':'city-L/100km'}, inplace=True)

# same for another column. A function could have been created
main_df["highway-mpg"] = 235 / main_df["highway-mpg"]
main_df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)

"""DATA NORMALIZATION"""

main_df['length'] = main_df['length']/main_df['length'].max()  # simple feature scaling [0,1]
main_df['height'] = main_df['height']/main_df['height'].max()  # simple feature scaling [0,1]
main_df['width'] = main_df['width']/main_df['width'].max()  # simple feature scaling [0,1]


"""# how to one-hot encode the fuel type column (turning categorical values into numerical ones)
dummy = pd.get_dummies(main_df['fuel-type'])  # creates a dummy object
main_df = pd.concat([main_df, dummy], axis=1)  # concatenates the dummy object with the df

# Binning
bins = np.linspace(main_df['price'].min(),main_df['price'].max(),4) # creates np arrays with 4 equally spaced numbers
group_names = ['Low','Medium','High'] # list with the label names
main_df['price_binned'] = pd.cut(main_df['price'],bins,labels=group_names,include_lowest=True) # creates the new column
print(main_df)"""
