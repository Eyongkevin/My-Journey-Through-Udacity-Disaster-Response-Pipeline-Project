
# My Journey Through Udacity Disaster Response Pipeline Project
------------
Here, I will be expressing the challenges while implementing the project.

## ETL Pipeline Preparation

### 1 Loading csv with pandas
**a.** When loading csv files with pandas like `pd.read_csv`, you may see a `pandas.errors.DtypeWarning`. This may be because of mixed type in the dataset. Check this [doc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.errors.DtypeWarning.html) to understand more
>Pandas tries to figure out programatically the data type of each column (integer, float, boolean, string). In this case, pandas could not automatically figure out the data type. That is because some columns have more than one possible data types. In other words, this data is messy.

**b.** When loading csv files and you encounter the error `ParserError: Error tokenizing data. C error: Expected 3 fields in line 5, saw 63`. It means the following
- It says line 5 doesn't follow the same format as the first 4 rows. 
- the first 4 rows has 3 fields(columns) while the 5th row has 63 fields.

**NB** *When we encounter this, we can investigate the dataset by viewing the content using pure python `open()`,`readline()` as below
```
# this will print the first 5 rows
with open("data.csv") as f:
    i = 0
    while i < 5:
        print(f.readline())
        i +=1
```

Depending how the data is, you will notice that the dataset is not well formated, and the first 4 rows are not formated as required. In pandas, we can ignore these rows and focus on the well formated rows. This is done using `pd.read_csv()` with its parameter `skiprows` where we specify how many rows to skip.

**c.** In every dataset, we need to deal with missing values. It is always neccessary to identify rows with missing values and deal with them independentlly. The rows with missing values can be extracted with:

`df[df.isnull().any(axis=1)]`

**d.** In pandas, strings are called "object" dtypes. 

### 2. Convert category values to just numbers 0 or 1.
Here we are asked to convert all category values to just numbers 0 or 1. However. The statement is as below:
- Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.

However, there are some category columns which have (2) as their last character. So it need to be converted to (1).
```
# Some string has (2), for example `related-2`. So we change all 2s to 1s
for column in categories:
    # For each value, convert 2s to 1s
    categories[column] = categories[column].apply(lambda x: 1 if x==2 or x==1 else 0)
```

### 3. Replace categories column in df with new category columns
Under this section, we are asked to concatenate the original dataframe with the new `categories` dataframe

You will have to use `axis=1` so that the concatination will take place along the columns instead of the rows/index which is default.

`df = pd.concat([df,categories], axis=1)`


## ML Pipeline Preparation

### 1. load data from database

Here we have to load the data and chose our X and Y which stands for `Predictor` and `target`.
Our predictor should be the message and the target here are multiple targets. So it should be all the categories.

```
X = df.message.values
Y = df.iloc[:,4:].values
```
You will notice that `df.iloc[:,4:].values` is only posible if you had concatinated your dataset this way
`df = pd.concat([df,categories], axis=1)`  with `axis=1`. So that the concatination will arrange the data columns in order starting with `df` and follow by `categories`. So in the code `df.iloc[:,4:].values`, we extract from the 4th column which is where the category colums start.

### 2. Build a machine learning pipeline

A simple machine pipeline will look like this:
```
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(),n_jobs=1))
    ])
```
The most important things to notice are:
- When runing the project locally, use `n_jobs = 1` and not `-1` as it may lead to the error:`ValueError: UPDATEIFCOPY base is read-only`.
- We use a multi-output classification to enable multiple-target classification. And we should note that not all machine learning model should be used here. You can see a list of accepted model from [here](http://scikit-learn.org/stable/modules/multiclass.html#multiclass)

### 3. Improve your model
Here, we improve the model by using GridSearchCV to find the best parameter combination. To know the possible 
parameters you could tune for the pipeline, use the code `pipeline.get_params()`. parameters with `__` are the ones you can set.

**Important**: It is very important to note that gridSearchCV is very expensive, so you should tune the least posible parameters. 
