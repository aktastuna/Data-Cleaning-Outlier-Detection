pip install tqdm
!pip install missingno
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from tqdm import tqdm

train_dataset = pd.read_csv(r'C:\Users\tunahan.aktas\Desktop\thy_train_data/Assessment Train Data.csv', na_values = ['?'])
test_dataset = pd.read_csv(r'C:\Users\tunahan.aktas\Desktop\thy_train_data/Assessment Result File.csv', na_values = ['?'])

train_dataset.columns = pd.Series(train_dataset.columns).apply(lambda x: x.lower())
test_dataset.columns = pd.Series(test_dataset.columns).apply(lambda x: x.lower())

df = train_dataset.copy()
df.info()
df.shape
df.dtypes

class data_preview:

    def __init__(self, df):
        self.df = df
        self.unique_number = pd.DataFrame(columns = ['variable', 'unique_number'])
        self.question_mark = pd.DataFrame(columns = ['variable', 'question_mark_number'])
        self.categorical_unique = pd.DataFrame(columns = ['variable', 'number_of_unique'])
        self.null_percentage = []
        self.null_chart = self.df.isnull().sum()
        self.cat_unique_percantage = pd.DataFrame()

    def get_unique_number(self):
        self.unique_number = pd.DataFrame(columns = ['variable', 'unique_number'])
        for i in range(len(self.df.columns.to_list())):
            self.unique_number = self.unique_number.append({'variable': self.df.columns[i],
                                                      'unique_number': self.df[self.df.columns[i]].nunique()}, 
    ignore_index = True)
        return self.unique_number

    def get_char_instead_of_null(self, character = '?'):
        self.char_null = pd.DataFrame(columns = ['variable', 'question_mark_number'])
        for i in range(len(self.df.columns.to_list())):
            if self.df[self.df.columns[i]].isin([character]).any():
                self.question_mark = self.question_mark.append({'variable': self.df.columns[i],
                                                           'question_mark_number': self.df[self.df[self.df.columns[i]].isin([character])][self.df.columns[i]].value_counts()[character]}, 
                ignore_index = True)
        else:
            self.char_null = self.char_null.append({'variable': self.df.columns[i], 'question_mark_number': 0}, ignore_index = True)
        return self.char_null

    def get_null_percentage(self, threshold = 0.8):
        self.null_percentage.clear()
        for i in range(len(self.null_chart.index.to_list())):
            if self.null_chart[i] / len(self.df) >= threshold:
                self.null_percentage.append(self.null_chart.index[i])
            else:
                continue
        return self.null_percentage

    def get_categorical_unique_number(self, threshold = 5):
        object_list = self.df.select_dtypes(include = 'object').columns
        self.categorical_unique = pd.DataFrame(columns = ['variable', 'number_of_unique'])
        for i in object_list:
            if self.df[i].nunique() > threshold:
                self.categorical_unique = self.categorical_unique.append({'variable': i,
                                                                      'number_of_unique': self.df[i].nunique()},
                    ignore_index  =True)
            else:
                continue
        return self.categorical_unique

    def get_categorical_unique_percantage(self, variable):
        self.cat_unique_percantage = pd.DataFrame()
        self.cat_unique_percantage = pd.DataFrame(self.df[variable].value_counts(dropna = False)).reset_index(drop = False)
        self.cat_unique_percantage['ratio'] = self.cat_unique_percantage[variable] / len(self.df)
        return self.cat_unique_percantage[['index', 'ratio']]
    
    def get_data_summary(self):
        return pd.concat([self.df.isnull().sum(), 
                          self.df.nunique(),
                          self.df.dtypes], axis = 1).rename(columns = {0: 'null_num', 1: 'num_level', 2: 'dtypes'})

df['departure_ymd_lmt'] = pd.to_datetime(df['departure_ymd_lmt'], format = '%Y%m%d', errors = 'coerce')
df['operation_ymd_lmt'] = pd.to_datetime(df['operation_ymd_lmt'], format = '%Y%m%d', errors = 'coerce')

oper_dict = {'JW': 'Online', 'TW': 'Online', 'TS': 'Mobile',
             'JM': 'Mobile', 'TY': 'Kontuar', 'QC': 'Kontuar', 'SC': 'Kiosk',
             'IR': 'Diger', '?': 'Diger', 'IA': 'Diger', 'BD': 'Diger',
             'CC': 'Diger', 'QR': 'Diger', 'QP': 'Diger', 'QA': 'Diger'}
df['operation_channel_group'] = df['operation_channel']
for i, j in oper_dict.items():
    df['operation_channel_group'] = df['operation_channel_group'].replace(i, j)

df = df.drop(columns = 'operation_channel', axis = 1)

df['passenger_gender'] = df['passenger_gender'].replace(['F/INF', 'M/INF', 'C/INF'], ['M', 'F', 'C'])

df[(pd.isnull(df['passenger_gender'])) & (df['passenger_title'] == 'MISTER')].loc[:, 'passenger_gender'] = 'M'
df['passenger_gender'].value_counts(dropna = False)

df['passenger_gender'] = np.where((df['passenger_title'] == 'MISTER') & (df['passenger_gender'].isnull()), 'M', 
    np.where((df['passenger_title'] == 'MISS') & (df['passenger_gender'].isnull()), 'F', 
             np.where((df['passenger_title'] == 'MISSES') & (df['passenger_gender'].isnull()), 'F', df['passenger_gender'])))

df['passenger_gender'] = np.where((df['passenger_title'] == 'MISTER') & (df['passenger_gender'] == 'F'), 'M', 
    np.where((df['passenger_title'] == 'MISS') & (df['passenger_gender'] == 'M'), 'F', 
             np.where((df['passenger_title'] == 'MISSES') & (df['passenger_gender'] == 'M'), 'F', df['passenger_gender'])))

my_class = data_preview(df)

# =============================================================================
# 1) Data Cleaning
# =============================================================================

## 1.1) Check variables has only 1 level.

unique_num_df = my_class.get_unique_number()
unique_num_list = unique_num_df[unique_num_df['unique_number'] == 1]['variable'].to_list()
df = df.drop(columns = unique_num_list, axis = 1)

## 1.2) Check variables contain at least %80 missing values.

my_class = data_preview(df)
delete_extreme_null = my_class.get_null_percentage(threshold = 0.75)
df = df.drop(columns = delete_extreme_null, axis = 1)

## 1.3) Check variables contain more than 5 levels.

my_class = data_preview(df)
categorical_num_df = my_class.get_categorical_unique_number(threshold = 10)
categorical_num_list = categorical_num_df[[categorical_num_df.columns[0]]][categorical_num_df.columns[0]].to_list()

my_class.get_categorical_unique_percantage(categorical_num_list[0])
delete_these_columns = categorical_num_list[0]

df = df.drop(delete_these_columns, axis = 1)

## 1.4) Category Level Reduction

my_class = data_preview(df)
categorical_num_df = my_class.get_categorical_unique_number(threshold = 5)
categorical_num_list = categorical_num_df[[categorical_num_df.columns[0]]][categorical_num_df.columns[0]].to_list()

change_this_variable = my_class.get_categorical_unique_percantage(categorical_num_list[1])
replace_these_values = change_this_variable[change_this_variable['ratio'] < .05]['index'].to_list()

df[categorical_num_list[1]] = df[categorical_num_list[1]].replace(replace_these_values, 'Other')

# terminal_name is having too many levels. Therefore, I will drop this column.
df = df.drop(columns = 'terminal_name', axis = 1)

## 1.5) Check missing values.

my_class = data_preview(df)
check = my_class.get_data_summary()

# I only have missing values in categorical variables.

for i in check['dtypes'].unique():
    print(check[check['dtypes'] == i])

############# CONTINUE HERE....

# =============================================================================
# 2) Outlier Detection
# =============================================================================

## 2.1) Simple Variable Outlier Detection

numeric_variables = df.select_dtypes(exclude = ['datetime64[ns]', 'object'])
cont_variables = []
for i in numeric_variables.columns.to_list():
    if numeric_variables[i].nunique() > 2:
        cont_variables.append(i)
cont_variables = df[cont_variables]

from collections import Counter

def detect_outliers(dataframe, variables):
    
    outlier_indices = []
    
    for i in variables:
        
        q1 = np.percentile(df[i], 25)
        q3 = np.percentile(df[i], 75)
        iqr = q3 - q1
        outlier_step = iqr * 1.5
        outlier_list_cols = df[(df[i] < q1 - outlier_step) | (df[i] > q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_cols)
    
    outlier_indices = Counter(outlier_list_cols)
    multiple_outliers = [i for i, v in outlier_indices.items() if v > 2]
    return multiple_outliers

df.loc[detect_outliers(numeric_variables, numeric_variables.columns.to_list())]

# There is no outlier in our data.

## 2.2) Multiple Variables Outlier Detection (LOF)

# NOF = Negative Outlier Factor
# Birkaç tane eşik değer bulabilirsin. NOF skorlarına bakarak birkaç tane eşik değer saptadıysan aşağıdaki konuları atlama!
# a) Eşik değerleri check etmeyi unutma. Karar verdiğin eşik değerdeki gözlemi incele. Gerçekten bir aykırılık söz konusu mu?
# b) Belirlenen outlier'ı eşik olarak aldığında kaç aykırı gözlem elde ediyorsun?
# b.1) İlgili eşik değeri ile çok fazla aykırı gözlem bulursan, çok veri kaybedersin. Bu konuyu göz ardı etme!
# Yukarıdaki maddeler doğrultusunda karar ver!

cont_variables.head()
from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)
clf.fit_predict(numeric_variables)
cont_variables_scores = np.sort(clf.negative_outlier_factor_)

check = pd.DataFrame(cont_variables_scores).rename(columns = {0: 'scores'})
check['substract'] = -(check['scores'] - check['scores'].shift())

esik_listesi = check[check['substract'] != 0]
esik_indices = esik_listesi.index

list_of_outliers = []
for i in tqdm(esik_indices):
    temp_esik_deger = cont_variables_scores[i]
    temp_aykiri_tf = cont_variables_scores > temp_esik_deger
    temp_new_df = cont_variables[cont_variables_scores <= temp_esik_deger]
    list_of_outliers.append([i, len(temp_new_df)])

esik_deger = cont_variables_scores[esik_indices[2077]]
aykiri_tf = cont_variables_scores > esik_deger
new_df = cont_variables[cont_variables_scores > esik_deger]

baski_deger = df[cont_variables_scores == esik_deger][0:1]
aykirilar = df[~aykiri_tf]
res = aykirilar.to_records(index = False)
res[:] = baski_deger.to_records(index = False)

df[~aykiri_tf] = pd.DataFrame(res, index = df[~aykiri_tf].index)

# =============================================================================
# 3) Missing Values
# =============================================================================

my_class.get_data_summary()
my_class.get_categorical_unique_percantage('outbound_arrival_airport')
df = df.drop(columns = 'outbound_arrival_airport', axis = 1)

# I only have missing values in categorical variables.

msno.bar(df.drop(columns = df.columns.to_list()[0:2]));
msno.matrix(df.drop(columns = df.columns.to_list()[0:2]));
msno.heatmap(df.drop(columns = df.columns.to_list()[0:2]));
# When we have missing values in gender, we do have missing value in title variables. There is a pattern here.

my_tab = pd.crosstab(index = [df["passenger_title"], df["cabin_class"]],  # Make a crosstab
                              columns="count")      # Name the count column
my_tab.plot.bar()

df['cabin_class'] = df['cabin_class'].fillna(df['cabin_class'].mode()[0])

df[df['passenger_title'].isnull()]['passenger_gender'].value_counts(dropna = False)
df[df['passenger_gender'].isnull()]['passenger_title'].value_counts(dropna = False)

df['passenger_title'] = np.where((df['passenger_gender'] == 'M') & (df['passenger_title'].isnull()), 
                                    'MISTER', df['passenger_title'])


df['operation_channel_group'] = df['operation_channel_group'].fillna(df['operation_channel_group'].mode()[0])
df = df.dropna()
my_class = data_preview(df)
my_class.get_data_summary()

dummies = pd.get_dummies(df.select_dtypes(include = 'object'))
df = df.drop(columns = df.select_dtypes(include = 'object').columns.to_list(), axis = 1)
df = pd.concat([df, dummies], axis = 1)
df = df.drop(columns = df.columns.to_list()[0:2])
