import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


"""PRINTING SETTINGS FOR DATAFRAMES AND ARRAYS"""

"""these instructions allow to see the all the columns of the dataframe on the run terminal.
To print all the rows use the method: print(main_df.to_string())"""

desired_width = 500
pd.set_option('display.width', desired_width)  # display width
pd.set_option('display.max_columns', 30)  # number of columns
"""np.set_printoptions(linewidth=desired_width)"""  # numpy set display width

"""MODULE 1"""
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
"""print(main_df)"""  # to print the first and the last 30 rows

"""MODULE 2"""
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

Col_mean_val = ['normalized-losses', 'bore', 'stroke', 'horsepower', 'peak-rpm']

# replaces the NaN values in all the columns in Col_mean_val with the mean of the column
for column in Col_mean_val:
    mean = main_df[column].astype("float").mean(axis=0)
    main_df[column].replace(np.nan, mean, inplace=True)

"""REPLACE MISSING VALUE WITH FREQUENCY"""

"""first method
mode = main_df['num-of-doors'].mode()  # returns a pandas series with one element, the mode
main_df['num-of-doors'].replace(np.nan, mode[0],inplace=True) # replaces NaN with the only element of the series mode
"""

"""second method"""
mode = main_df['num-of-doors'].value_counts().idxmax()  # value_counts for counting unique values. idxmax for mode
main_df['num-of-doors'].replace(np.nan, mode, inplace=True)  # replaces NaN with the only element of the series mode

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
main_df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)

# same for another column. A function could have been created
"""main_df["highway-mpg"] = 235 / main_df["highway-mpg"]
main_df.rename(columns={'highway-mpg': 'highway-L/100km'}, inplace=True)"""

"""DATA NORMALIZATION"""

main_df['length'] = main_df['length'] / main_df['length'].max()  # simple feature scaling [0,1]
main_df['height'] = main_df['height'] / main_df['height'].max()  # simple feature scaling [0,1]
main_df['width'] = main_df['width'] / main_df['width'].max()  # simple feature scaling [0,1]

"""DATA BINNING"""

# Firstly lets plot the histogram of the column to see how is the distribution
"""plt.hist(main_df["horsepower"])  # plot histogram. If we add bins = 3 it shows the histogram on 3 bins
plt.xlabel("horsepower")  # set x label
plt.ylabel("count")  # set y label
plt.title("horsepower bins")  # plot title
plt.show() #  command to open a new window and show the plot"""

"""Cut the column in equally spaced bins"""

# creates np arrays with 4 equally spaced numbers
bins = np.linspace(main_df['horsepower'].min(), main_df['horsepower'].max(), 4)

# list with the label names
group_names = ['Low', 'Medium', 'High']

# creates the new column
main_df['horsepower_binned'] = pd.cut(main_df['horsepower'], bins, labels=group_names, include_lowest=True)

"""INDICATOR (OR DUMMY) VARIABLES / HOT-ONE ENCODING"""

"""that's how to turn categorical values into numerical values"""

dummy_var1 = pd.get_dummies(main_df['fuel-type'])  # creates a dummy object
dummy_var1.rename(columns={'fuel-type-gas': 'gas', 'fuel-type-diesel': 'diesel'}, inplace=True)  # renames columns
main_df = pd.concat([main_df, dummy_var1], axis=1)  # concatenates the dummy object with the df
main_df.drop("fuel-type", axis=1, inplace=True)  # drops the original column

dummy_var2 = pd.get_dummies(main_df['aspiration'])
dummy_var2.rename(columns={'std': 'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)
main_df = pd.concat([main_df, dummy_var2], axis=1)
main_df.drop("aspiration", axis=1, inplace=True)  # drops the original column

"""MODULE 3"""
"""DESCRIPTIVE STATISTICS"""

# to compute and show different statistics of the numerical columns
"""print(main_df.describe())"""

# to include also the object type in the table
"""print(main_df.describe(include=['object]'))"""

# another good descriptor of categorical values is the value counts. Value counts only takes Pandas series
# so we have single [] and not double [[]]
"""print(main_df['engine-location'].value_counts())"""

# we can print a refined version of the df value counts
"""engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
print(engine_loc_counts)"""

# to show the Pearson coefficient of correlation between all the numerical columns
"""print(main_df.corr())  # useful to spot high correlations"""

# to investigate further a correlation between two numerical variables we can plot the scatterplot
# with the regression line
"""width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot('engine-size','price', data=main_df)
plt.ylim(0,)
plt.show()"""

# and then we calculate the correlation between the two columns
"""main_df[['engine-size','price']].corr()"""

# for categorical value we use the boxplot to investigate correlations
"""sns.boxplot(x="drive-wheels", y="price", data=main_df)
plt.show()"""

"""GROUPING"""

# to look at how many categories there are in a variable
"""print(main_df['drive-wheels'].unique())"""

# to group by those unique values
"""df_group_one = main_df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
print(df_group_one)"""

# to group by two categorical values
"""df_group_one = df_group_one.groupby(['drive-wheels', 'body-style'],as_index=False).mean()"""

# to make a pivot out of the grouped variable
"""grouped_pivot = df_group_one.pivot(index='drive-wheels',columns='body-style')
# to fill the NaN values of the pivot (when a combination is not present in the dataset) with a 0
grouped_pivot = grouped_pivot.fillna(0)"""

# to create a heat-map of the pivot table
"""plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()

# to change labels visualization in the heat map
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')
#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index
#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)
#rotate label if too long
plt.xticks(rotation=90)
# show heat map
fig.colorbar(im)
plt.show()"""

"""CORRELATION"""

# Pearson correlation coefficient and P-value
"""pearson_coef, p_value = stats.pearsonr(main_df['wheel-base'], main_df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)"""

"""ANOVA"""

# Firstly we need to group by a variable
"""grouped_test2= main_df[['drive-wheels', 'price']].groupby(['drive-wheels'])"""

# we can get one group with the command get_group
"""print(grouped_test2.get_group('4wd')['price'])"""

# to calculate the F-test score and the p-value
"""f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'],
                              grouped_test2.get_group('4wd')['price'])

print("ANOVA results: F=", f_val, ", P =", p_val)  """

"""MODULE 4"""
"""MODEL DEVELOPMENT"""

"""Single linear regression (Scikit-Learn)"""
"""
lm_slr = LinearRegression()  # Creates a LinearRegression object
X = main_df[['highway-mpg']]  # define the predictor
lm_slr.fit(X, main_df['price'])  # Train the module
Yhat_slr = lm_slr.predict(X)  # Predict target variable
print(Yhat[0:5])  # show target value
print(lm.intercept_)  # show the intercept of the regression line
print(lm.coef_)  # show the slope of the regression line"""

"""Multiple linear regression(Scikit-Learn)"""
"""
lm_mlr = LinearRegression()  # Creates a LinearRegression object
Z = main_df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]  # define the predictors
lm_mlr.fit(Z, main_df['price'])  # Train the module
Yhat_mlr = lm_mlr.predict(Z)  # predicts target variable"""

"""Model visualization"""
"""
# regression line
sns.regplot(main_df[['highway-mpg']],main_df['price'])

# residual plot (mainly for single linear regression)
sns.residplot(main_df[['highway-mpg']],main_df['price'])

# Distribution plot (mainly for multiple linear regression)
ax1 = sns.distplot(main_df['price'], hist=False, color='r', label='actual value')
sns.distplot(Yhat_mlr, hist=False, color='b', label='predicted value', ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')
plt.close()"""

"""Polynomial regression (Numpy or Scikitlearn)"""
"""
# numpy module for single polynomial regression (it works with SLR setting degree=1 in the polyfit method)
x = main_df['highway-mpg']
y = main_df['price']
f = np.polyfit(x, y, 3)  # train the module with a polynomial degree=3
p = np.poly1d(f)  # p is the polynomial function
print(p)  # displays the function
print(p(30))  # to make and display predictions"""

# function to plot polynomial regression
"""
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

PlotPolly(p, x, y, 'highway-mpg')"""

"""Multivariate Polynomial regression (Scikitlearn)"""
"""
pr = PolynomialFeatures(degree=2, include_bias=False)  # creates a PolynomialFeatures object
Z_pr = pr.fit_transform(Z)  # creates a new set of predictor with the combinations of the original predictors
lm_pr = LinearRegression()
lm_pr.fit(Z_pr, main_df['price'])  # Train the module
Yhat_pr = lm_pr.predict(Z_pr)  # predict target variable

# draw a distribution plot
ax1 = sns.distplot(main_df['price'], hist=False, color='r', label='actual value')
sns.distplot(Yhat_pr, hist=False, color='b', label='predicted value', ax=ax1)
plt.show()
plt.close()"""

"""Pipelines (Scikitlearn) Example for a Multivariate polynomial regression"""
"""
# firstly we create the pipeline
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]

# we input the list to a pipeline constructor
pipe=Pipeline(Input)

# perform all the actions in the pipeline
pipe.fit(Z,main_df['price'])"""

"""MODEL NUMERICAL EVALUATION"""

"""Single linear regression"""
"""
# Mean Squared Error (MSE)
mse = mean_squared_error(main_df['price'], Yhat_slr)
print('<SLR> The mean square error of price and predicted value is: ', mse)
# R^2
print(lm_slr.score(X,main_df['price']))"""

"""Multiple linear regression"""
"""
# Mean Squared Error (MSE)
mse = mean_squared_error(main_df['price'], Yhat_mlr)
print('<MLR> The mean square error of price and predicted value is: ', mse)
# R^2
print(lm_mlr.score(Z,main_df['price']))"""

"""Polynomial regression"""
"""
# Mean Squared Error (MSE)
mse = mean_squared_error(main_df['price'], p(x))
print('<PR> The mean square error of price and predicted value is: ', mse)

# R^2 (we are using Numpy so it's a different code)
r_squared = r2_score(main_df['price'], p(x))
print('The R-square value is: ', r_squared)"""

"""Multivariate Polynomial regression"""
"""
# Mean Squared Error (MSE)
mse = mean_squared_error(main_df['price'], Yhat_pr)
print('<MPR> The mean square error of price and predicted value is: ', mse)
# R^2
print(lm_pr.score(Z_pr,main_df['price']))"""

"""MODEL EVALUATION AND REFINEMENT"""

# Separate predictors from target
y_data = main_df['price']
x_data = main_df.drop('price', axis=1)

# split the data set in training and testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)

# linear regression model on one predictor
lre = LinearRegression()
lre.fit(x_train[['horsepower']], y_train)
print("number of test samples :", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

# scoring the linear regression model
print("train score: ", lre.score(x_train[['horsepower']], y_train))
print("test score: ", lre.score(x_test[['horsepower']], y_test))

"""Cross Validation (K-fold)"""

# Cross validation on 4 folds. Returns the array of scores i.e. R^2
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())

# Predictions with cross validation
Yhat_cr = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4)
print("Prediction with cross validation: ", Yhat_cr[0:5])

"""Overfitting, Underfitting and Model Selection"""


# Function for distribution plot (only formatting)
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

# Second function for plotting polinomial regression (for test and train datasets + formatting)
def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data
    # testing data
    # lr:  linear regression object
    # poly_transform:  polynomial transformation object

    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])
    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

# Create another Linear regression model
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# prediction with train dataset
Yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print("Prediction with train dataset: ", Yhat_train[0:5])

# Prediction with test dataset
Yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print("Prediction with test dataset: ", Yhat_test[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, Yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
# plt.show()
plt.close()

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,Yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
# plt.show()
plt.close()


# Create a 5th degree polynomial transformation of the variable 'horsepower'
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])

# Crate a polynomial model with the already transformed variable
poly = LinearRegression()
poly.fit(x_train_pr, y_train)
yhat = poly.predict(x_test_pr)

PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly,pr)
#plt.show()
plt.close()

# calculate the score for test and train
print("train score: ", poly.score(x_train_pr, y_train))
print("test score: ", poly.score(x_test_pr, y_test))

# calculate the score for different order polynomial
Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
#plt.show()
plt.close()

# for interactive display in jupiter notebook
"""
from IPython.display import display
#from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

%%capture
! pip install ipywidgets
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train,y_test, poly, pr)
    plt.show()

interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))"""
