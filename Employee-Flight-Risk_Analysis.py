# imports
import pandas as pd
import os
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, fbeta_score, recall_score, plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

# Load the data
df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
# drop irrelevant columns
df.drop(['EmployeeNumber', 'EmployeeCount', 'StandardHours', 'Over18', 'JobLevel'],
        axis=1, inplace=True)

# ## Pre-processing
#
#
# At a high level, this involves:
# 1. Encoding Target column with pandas.get_dummies()
# 2. One-hot encoding the nominal features using sklearn.OneHotEncoder
# 3. Scale all features to lie between $[0, 1]$
# 4. After running a GridSearchCV with RandomForestClassifer (not included here),
#    drop 17 lowest-ranked features.

# #### 1. Dropping numerical columns used for plotting
Attrition_enc = pd.get_dummies(df['Attrition'], drop_first=True)
Attrition_enc.columns = ['Attrition-enc']

# #### 3. One-hot encoding the nominal features using sklearn.OneHotEncoder

# First separate the target and feature matrix
X = df.drop('Attrition', axis='columns')
y = Attrition_enc['Attrition-enc']

# Now apply OneHotEncoder to the nominal columns using ColumnTransformer

# get nominal categorical columns in list
cat_cols = X.columns[X.dtypes == object].tolist()

# set up the ColumnTransformer and OneHotEncoder
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(sparse=False, drop='first'), cat_cols)],
    remainder='passthrough'
)

# set up column names for dummy columns which will be used when we create a new X dataframe
dummies = ['BusinessTravel_Travel_Frequently',
           'BusinessTravel_Travel_Rarely',
           'Department_Research & Development',
           'Department_Sales',
           'EducationField_Life Sciences',
           'EducationField_Marketing',
           'EducationField_Medical',
           'EducationField_Other',
           'EducationField_Technical Degree',
           'Gender_Female',
           'JobRole_Human Resources',
           'JobRole_Laboratory Technician',
           'JobRole_Manager',
           'JobRole_Manufacturing Director',
           'JobRole_Research Director',
           'JobRole_Research Scientist',
           'JobRole_Sales Executive',
           'JobRole_Sales Representative',
           'MaritalStatus_Married',
           'MaritalStatus_Single',
           'OverTime_Yes']

# Now let's get the full column list by appending dummies to the original columns list of X minus
# the categorical columns themselves (since we are going to use one-hot-encoded columns instead)
new_cols = dummies + list(X.columns)
new_cols = [e for e in new_cols if e not in cat_cols]

# now we apply the ColumnTransformer to X and create a new dataframe
X = pd.DataFrame(ct.fit_transform(X), columns=new_cols)

# #### 4. Scale all features to lie between \[0, 1]

# We know the dummy features are nominal and are either 0 or 1 already so we do not need to
# normalize them. However, all other features including the ordinal categorical variables will be
# scaled to lie in the interval $[0, 1]$

# Get all non-dummy columns
cols_non_dummy = [col for col in list(X.columns) if col not in dummies]
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X[cols_non_dummy])
scaled_arr = scaler.transform(X[cols_non_dummy])
X_scaled = pd.DataFrame(scaled_arr, columns=cols_non_dummy)

# Now get the rest of the X dataframe appended
X_scaled = pd.concat([X.drop(cols_non_dummy, axis=1), X_scaled], axis=1)

# 4. After running a GridSearchCV with RandomForestClassifer (not included here),
#    drop 17 lowest-ranked features. I didn't include the GridSearch as it was very long to run.

cols2drop = ['WorkLifeBalance',
             'MaritalStatus_Married',
             'JobRole_Laboratory Technician',
             'Education',
             'EducationField_Technical Degree',
             'EducationField_Medical',
             'EducationField_Life Sciences',
             'BusinessTravel_Travel_Rarely',
             'JobRole_Research Scientist',
             'BusinessTravel_Non-Travel',
             'JobRole_Manufacturing Director',
             'EducationField_Marketing',
             'PerformanceRating',
             'JobRole_Manager',
             'EducationField_Other',
             'JobRole_Human Resources',
             'JobRole_Research Director']

# Set up cols to keep
cols2keep = [i for i in X_scaled.columns if i not in cols2drop]

# drop from dataset
X_red = X_scaled[cols2keep]  # this will leave 27 columns in X_Red

### Model with RandomForestClassifier

# test-train split
X_train, X_test, y_train, y_test = train_test_split(X_red, y, test_size=0.25, random_state=666)

# parameters to use (found with GridSearchCV - not included in script due to long run-time)
rf_params = {'class_weight': {0: 0.12, 1: 0.88}, 'max_depth': 10,
             'max_features': 0.15, 'min_samples_leaf': 20,
             'min_samples_split': 10, 'min_weight_fraction_leaf': 0.001,
             'n_estimators': 120, 'n_jobs': -1, 'random_state': 666}

# train and fit
rf = RandomForestClassifier(**rf_params)  # use dict unpacking
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# string to print
buffer_str = '\n'

# get score results
fbeta = fbeta_score(y_test, y_pred, beta=1.75)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# now append the actual results to the output string
buffer_str += f'F-beta: {fbeta:0.3f}\n'
buffer_str += '-' * 40 + '\n'
buffer_str += f'recall: {recall:0.3f}\n'
buffer_str += '-' * 40 + '\n\n'
buffer_str += f'Classification Report:\n{report}\n\n'
buffer_str += '=' * 80 + '\n\n'

# print
print(buffer_str)

### Prepare performance plots

# confusion matrix
class_names = sorted(df['Attrition'].unique())
np.set_printoptions(precision=3)
disp = plot_confusion_matrix(rf, X_test, y_test,
                             display_labels=class_names,
                             cmap=plt.cm.Greens,
                             normalize=None,
                             values_format='.3g')
disp.ax_.set_title("Confusion matrix")
plt.show()

# print confusion matrix
print("Confusion matrix")
print(disp.confusion_matrix)

# Save to file
os.makedirs('plots', exist_ok=True)
disp.figure_.savefig('plots/ConfusionMatrix.png')

# Precision-Recall Curve
# noinspection PyTypeChecker
disp = plot_precision_recall_curve(rf, X_test, y_test, color='g')
x_data = disp.line_.get_xdata()
y_data = disp.line_.get_ydata()
disp.ax_.fill_between(x_data, y_data, alpha=0.25, color='g')
disp.ax_.set_title(f'Precision-Recall curve')
plt.show()

# Save to file
os.makedirs('plots', exist_ok=True)
disp.figure_.savefig('plots/Precision-Recall_curve.png')

# Feature Importance Chart
# start by initializing a dataframe to hold results
feature_importance_df = pd.DataFrame(columns=['Feature', 'Importance'])
feature_importance_df['Feature'] = X_train.columns.values
feature_importance_df['Importance'] = rf.feature_importances_

# sort by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
feature_importance_df.reset_index(drop=True, inplace=True)

y_ticks = sorted(range(27), reverse=True)
fig, ax = plt.subplots()
ax.barh(y_ticks, feature_importance_df['Importance'], color='g')
ax.set_yticklabels(feature_importance_df['Feature'])
ax.set_yticks(y_ticks)
ax.set_title("RF Feature Importance in explaining Flight Risk")
fig.tight_layout()
plt.show()

# Save to file
os.makedirs('plots', exist_ok=True)
fig.savefig('plots/RF_FeatureRank.png')
