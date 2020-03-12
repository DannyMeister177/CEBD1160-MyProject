import pandas as pd

# Load the data
df = pd.read_csv('data/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.drop('EmployeeCount', axis=1, inplace=True)        # Column has no relevant info

# Print info and summary stats
print(f'Info and description of dataset:\n')
print(df.info())
print(f'------------------------------------------------------------\n\n')
print(f'Print descriptive statistics.\n')
print(df.describe().dropna(how='all').to_string())
print(f'------------------------------------------------------------\n\n')

# Print correlations
print(f'Printing the correlations.\n')
print(df.corr().dropna(how='all').to_string())
print(f'------------------------------------------------------------\n\n')
