import pandas as pd

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
print(main_df.head())


