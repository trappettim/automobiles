import pandas as pd

"""PRINTING SETTINGS FOR DATAFRAMES AND ARRAYS"""
"""these instructions allow to see the all the columns of the dataframe on the run terminal.
To print all the rows use the method: print(main_df.to_string())"""

desired_width = 320
pd.set_option('display.width', desired_width) #display width
pd.set_option('display.max_columns',30) #number of columns
#np.set_printoptions(linewidth=desired_width) #numpy set display width

"""IMPORTING DATASET INTO PANDAS DATAFRAME"""

url = 'imports-85.data'
headers_names = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
"num-of-doors", "body-style", "drive-wheels", "engine-location",
"wheel-base", "length", "width", "height", "curb-weight",
"engine-type", "num-of-cylinders", "engine-size", "fuel-system",
"bore", "stroke", "compression-ratio", "hoursepower", "peak-rpm",
"city-mpg", "highway-mpg", "price"]
# read csv, no headers in the file, put headers_names as headers, missing values identified by '?'
main_df = pd.read_csv(url,header=None,names=headers_names,na_values='?')
print(main_df.to_string())
pass


