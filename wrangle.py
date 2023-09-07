"""
Wrangle Zillow Data

Functions:
- wrangle_zillow_mvp
- wrangle_zillow
    - get_zillow
    - prep4ex_zillow
- split_zillow
- mm_zillow
- std_zillow
- robs_zillow
- encode_county
"""
"""
Occasional setting change

Default is 60, use none to see the whole thing

pd.set_option('display.max_rows', None)
"""

### IMPORTS ###

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from env import user, password, host

### ACQUIRE DATA ###


def get_zillow(user=user, password=password, host=host):
    """
    This function acquires data from a SQL database of 2017 Zillow properties and caches it locally.

    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `aq_zillow` is returning a dirty pandas DataFrame
    containing information on single family residential properties
    """
    # name of cached csv
    filename = "zillow.csv"
    # if cached data exist
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    # wrangle from sql db if not cached
    else:
        # read sql query into df
        # 261 is single family residential id
        df = pd.read_sql(
            """select * 
                            from properties_2017 
                            left join predictions_2017 using(parcelid) 
                            where propertylandusetypeid in (261,279)""",
            f"mysql+pymysql://{user}:{password}@{host}/zillow",
        )
        # filter to just 2017 transactions
        df = df[df["transactiondate"].str.startswith("2017", na=False)]
        # cache data locally
        df.to_csv(filename, index=False)
    return df


def explore_prep_zillow(df):
    """send uncleaned zillow df to prep for exploration"""
    # replace missing values with "0" or appropriate value where it makes sense
    df = df.fillna(
        {
            "numberofstories": 0,
            "fireplaceflag": 0,
            "yardbuildingsqft26": 0,
            "yardbuildingsqft17": 0,
            "unitcnt": 0,
            "threequarterbathnbr": 0,
            "pooltypeid7": 0,
            "pooltypeid2": 0,
            "pooltypeid10": 0,
            "poolsizesum": 0,
            "poolcnt": 0,
            "hashottuborspa": 0,
            "garagetotalsqft": 0,
            "garagecarcnt": 0,
            "fireplacecnt": 0,
            "lotsizesquarefeet": df["calculatedfinishedsquarefeet"],
        }
    )
    # split transaction date to year, month, and day
    df_split = df["transactiondate"].str.split(pat="-", expand=True).add_prefix("trx_")
    df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)
    # rename columns
    df = df.rename(
        columns=(
            {
                "yearbuilt": "year",
                "bedroomcnt": "beds",
                "bathroomcnt": "baths",
                "calculatedfinishedsquarefeet": "area",
                "taxvaluedollarcnt": "prop_value",
                "fips": "county",
                "trx_1": "trx_month",
                "trx_2": "trx_day",
                "numberofstories": "stories",
                "poolcnt": "pools",
            }
        )
    )
    # filter out/drop columns that have too many nulls, are related to target, are dupes, or have no use for exploration or modeling
    df = df.drop(
        columns=[
            "id",
            "airconditioningtypeid",
            "architecturalstyletypeid",
            "basementsqft",
            "buildingclasstypeid",
            "buildingqualitytypeid",
            "calculatedbathnbr",
            "decktypeid",
            "finishedfloor1squarefeet",
            "finishedsquarefeet12",
            "finishedsquarefeet13",
            "finishedsquarefeet15",
            "finishedsquarefeet50",
            "finishedsquarefeet6",
            "fullbathcnt",
            "heatingorsystemtypeid",
            "lotsizesquarefeet",
            "pooltypeid10",
            "pooltypeid2",
            "pooltypeid7",
            "propertycountylandusecode",
            "propertylandusetypeid",
            "propertyzoningdesc",
            "rawcensustractandblock",
            "regionidcity",
            "regionidcounty",
            "regionidneighborhood",
            "regionidzip",
            "storytypeid",
            "threequarterbathnbr",
            "typeconstructiontypeid",
            "yardbuildingsqft17",
            "yardbuildingsqft26",
            "structuretaxvaluedollarcnt",
            "assessmentyear",
            "landtaxvaluedollarcnt",
            "taxamount",
            "taxdelinquencyflag",
            "taxdelinquencyyear",
            "censustractandblock",
            "id.1",
            "logerror",
        ]
    )
    # drop nulls
    df = df.dropna()
    # map county to fips
    df.county = df.county.map({6037: "LA", 6059: "Orange", 6111: "Ventura"})
    # make int
    ints = ["year", "beds", "area", "prop_value", "trx_month", "trx_day"]
    for i in ints:
        df[i] = df[i].astype(int)
    # sort by column: 'transactiondate' (descending) for dropping dupes keeping recent
    df = df.sort_values(["transactiondate"], ascending=[False])
    # drop duplicate rows in column: 'parcelid', keeping max trx date
    df = df.drop_duplicates(subset=["parcelid"])
    # add features
    df = df.assign(age=2017 - df.year)
    # then sort columns and index for my own eyes
    df = df[
        [
            "age",
            "baths",
            "beds",
            "roomcnt",
            "area",
            "county",
            "latitude",
            "longitude",
            "prop_value",
        ]
    ].sort_index()
    # drop outlier rows based on column: 'prop_value' and 'area'
    df = df[(df["prop_value"] < df["prop_value"].quantile(0.98)) & (df["area"] < 6000)]
    return df


def wrangle_zillow_mvp():
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.

    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, property value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    df = get_zillow()
    df = prep4ex_zillow(df)
    return df[["beds", "baths", "area", "prop_value"]].assign(
        rooms=(df.beds + df.baths)
    )


def wrangle_zillow():
    """
    This function wrangles data from a SQL database of Zillow properties, caches it locally, drops null
    values, renames columns, maps county to fips, converts certain columns to integers, and handles
    outliers.

    :param user: The username for accessing the MySQL database
    :param password: The password is unique per user saved in env
    :param host: The host parameter is the address of the server where the Zillow database is hosted
    :return: The function `wrangle_zillow` is returning a cleaned and wrangled pandas DataFrame
    containing information on single family residential properties in Los Angeles, Orange, and Ventura
    counties, including the year built, number of bedrooms and bathrooms, square footage, property value,
    property tax, and county. The DataFrame has been cleaned by dropping null values, renaming columns,
    mapping county codes to county names, converting certain columns
    """
    df = get_zillow()
    df = prep4ex_zillow(df)
    return df


### SPLIT DATA ###


def split_data(df):
    """Split into train, validate, test with a 60/20/20 ratio"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=42)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=42)
    return train, validate, test


### SCALERS ###


def mm_zillow(train, validate, test, scale=None):
    """
    The function applies the Min Max Scaler method to scale the numerical features of the train, validate,
    and test datasets.

    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    mm_scale = MinMaxScaler()
    Xtr, Xv, Xt = train[scale], validate[scale], test[scale]
    Xtr = pd.DataFrame(mm_scale.fit_transform(train[scale]), train.index, scale)
    Xv = pd.DataFrame(mm_scale.transform(validate[scale]), validate.index, scale)
    Xt = pd.DataFrame(mm_scale.transform(test[scale]), test.index, scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f"{col}_s"})
        Xv = Xv.rename(columns={col: f"{col}_s"})
        Xt = Xt.rename(columns={col: f"{col}_s"})
    return Xtr, Xv, Xt


def std_zillow(train, validate, test, scale=None):
    """
    The function applies the Standard Scaler method to scale the numerical features of the train, validate,
    and test datasets.

    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    std_scale = StandardScaler()
    Xtr, Xv, Xt = train[scale], validate[scale], test[scale]
    Xtr = pd.DataFrame(std_scale.fit_transform(train[scale]), train.index, scale)
    Xv = pd.DataFrame(std_scale.transform(validate[scale]), validate.index, scale)
    Xt = pd.DataFrame(std_scale.transform(test[scale]), test.index, scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f"{col}_s"})
        Xv = Xv.rename(columns={col: f"{col}_s"})
        Xt = Xt.rename(columns={col: f"{col}_s"})
    return Xtr, Xv, Xt


def robs_zillow(train, validate, test, scale=None):
    """
    The function applies the RobustScaler method to scale the numerical features of the train, validate,
    and test datasets.

    :param train: a pandas DataFrame containing the training data
    :param validate: The validation dataset, which is used to evaluate the performance of the model
    during training and to tune hyperparameters
    :param test: The "test" parameter is a dataset that is used to evaluate the performance of a machine
    learning model that has been trained on the "train" dataset and validated on the "validate" dataset.
    The "test" dataset is typically used to simulate real-world scenarios and to ensure that the model
    is able
    :return: three dataframes: Xtr (scaled training data), Xv (scaled validation data), and Xt (scaled
    test data).
    """
    if scale is None:
        scale = train.columns.to_list()
    rob_scale = RobustScaler()
    Xtr, Xv, Xt = train[scale], validate[scale], test[scale]
    Xtr = pd.DataFrame(rob_scale.fit_transform(train[scale]), train.index, scale)
    Xv = pd.DataFrame(rob_scale.transform(validate[scale]), validate.index, scale)
    Xt = pd.DataFrame(rob_scale.transform(test[scale]), test.index, scale)
    for col in scale:
        Xtr = Xtr.rename(columns={col: f"{col}_s"})
        Xv = Xv.rename(columns={col: f"{col}_s"})
        Xt = Xt.rename(columns={col: f"{col}_s"})
    return Xtr, Xv, Xt


### ENCODE ###


def encode_county(df):
    """
    Encode county column from zillow dataset
    """
    df["Orange"] = df.county.map({"Orange": 1, "Ventura": 0, "LA": 0})
    df["LA"] = df.county.map({"Orange": 0, "Ventura": 0, "LA": 1})
    df["Ventura"] = df.county.map({"Orange": 0, "Ventura": 1, "LA": 0})
    return df


def check_columns(df_telco):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, the proportion of null values,
    and the data type of the column. The resulting dataframe is sorted by the
    'Number of Unique Values' column in ascending order.

    Args:
    - df_telco: pandas dataframe

    Returns:
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df_telco.columns:
        # Append the column name, number of unique values, unique values, number of null values, proportion of null values, and data type to the data list
        data.append(
            [
                column,
                df_telco[column].nunique(),
                df_telco[column].unique(),
                df_telco[column].isna().sum(),
                df_telco[column].isna().mean(),
                df_telco[column].dtype,
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', 'Proportion of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype",
        ],
    )
