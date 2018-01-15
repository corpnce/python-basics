import numpy as np
import pandas as pd
# Loading Data set
df = pd.read_csv('D:/version-control/my-projects/datasets/xAPI-Edu-Data.csv')
df.head()
print(df.shape)
df.isnull().sum()
# Visualising Data Set
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='Topic', data=df, palette='muted')
# plt.show()
df['Failed'] = np.where(df['Class']== 'L',1,0)
sns.factorplot(x='Topic', y='Failed', data=df, size=9)
pd.crosstab(df['Class'], df['Topic'])
sns.countplot(x='Class', data=df, palette='PuBu')
df.Class.value_counts()
Raised_hand = sns.boxplot(x="Class", y="raisedhands", data=df)
Raised_hand = sns.swarmplot(x="Class", y="raisedhands", data=df, color=".15")
# plt.show()
df.groupby('Topic').median()
df['Abs'] = df['StudentAbsenceDays']
df['Abs'] = np.where(df['Abs'] == 'Under-7', 0, 1)
df['Abs'].groupby(df['Topic']).mean()
# Creating dataset for classifer
x = np.array(df.raisedhands)
df['TotalQ'] = df['Class']
df['TotalQ'].loc[df.TotalQ == 'L'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'M'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'H'] = 2.0
y = np.array(df['TotalQ'])
# Visualizing X and Y

df['TotalQ'] = df['Class']

df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0
df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0
df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0

continuous_subset = df.ix[:,9:13]
continuous_subset['gender'] = np.where(df['gender']=='M',1,0)
continuous_subset['Parent'] = np.where(df['Relation']=='Father',1,0)

X = np.array(continuous_subset).astype('float64')
y = np.array(df['TotalQ'])
plt.scatter(X,y,label='skitscat', color='k')
plt.show()
