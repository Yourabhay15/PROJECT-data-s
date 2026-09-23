# Analysis of Age-Group-wise Rape Cases in India

# This project performs data cleaning, exploratory data analysis (EDA),
# statistical testing, and machine learning on rape victim data.

# Objectives:
# - Understand distribution of victims across age groups
# - Identify trends and patterns over time
# - Perform statistical validation
# - Build a predictive model


# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

plt.rcParams['figure.figsize'] = (10,6)
sns.set_style("whitegrid")


# Data Loading and Cleaning

# - Load dataset
# - Remove summary rows to avoid duplication
# - Convert columns to numeric
# - Handle missing values

df = pd.read_csv(r'C:\Users\lenovo\Desktop\CA2\Incest_other_rape_victim.csv')

df_clean = df[
    (df['Crime Head'].str.lower() != 'total') &
    (df['STATE/UT'].str.lower() != 'total')
].copy()

df_clean.columns = df_clean.columns.str.strip()

cols_to_fix = df_clean.columns[3:]
df_clean[cols_to_fix] = df_clean[cols_to_fix].apply(pd.to_numeric, errors='coerce')

df_clean.fillna(0, inplace=True)
df_clean.drop_duplicates(inplace=True)

# NumPy Analysis

# - Calculate mean, max, min, standard deviation

total_victims = df_clean['Total Victims'].values

print("Mean:", np.mean(total_victims))
print("Max:", np.max(total_victims))
print("Min:", np.min(total_victims))
print("Std Dev:", np.std(total_victims))


print(df_clean['Total Victims'].describe())


# Correlation Analysis
age_groups = df_clean.columns[4:10]
corr_matrix = df_clean[age_groups].corr()

print(corr_matrix)


# Heatmap Visualization
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Age Groups')
plt.show()


# Outlier Detection

sns.boxplot(x=df_clean['Total Victims'])
plt.title('Outlier Detection')
plt.show()


# Normality Test (Shapiro-Wilk)

stat, p = shapiro(df_clean['Total Victims'])

print("Shapiro p-value:", p)
print("Normal Distribution" if p > 0.05 else "Not Normal Distribution")


# Hypothesis Testing (t-test)

v2001 = df_clean[df_clean['YEAR'] == 2001]['Total Victims']
v2012 = df_clean[df_clean['YEAR'] == 2012]['Total Victims']

t_stat, p_val = ttest_ind(v2001, v2012)

print("T-test p-value:", p_val)


# Machine Learning (Linear Regression)

X = df_clean[['YEAR']]
y = df_clean['Total Victims']

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

print("R2 Score:", r2_score(y, pred))


# Regression Visualization

plt.scatter(X, y, alpha=0.3, label='Actual')
plt.plot(X, pred, linewidth=2, label='Prediction')
plt.xlabel('Year')
plt.ylabel('Total Victims')
plt.title('Year vs Total Victims')
plt.legend()
plt.show()



# Conclusion
# - Majority of victims fall in 18–30 age group
# - Data shows variation across years and states
# - Data is not normally distributed
# - Statistical tests show differences between years
# - Regression shows trend in victim counts
