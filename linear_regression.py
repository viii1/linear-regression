import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('/Users/vivek/Desktop/git/Ai/linera-regression/austin_weather.csv') 

print(dataset.shape)

# here the data set contains 1319 rows and 21 colums which means 21 PREDICTOR VARIABLES

print(dataset.describe())

dataset.plot(x='TempLowF', y='TempHighF', style='o')
plt.title('TempLowF vs TempHighF')
plt.xlabel('TempLowF')
#plt.ylable('TempHighF')

plt.show()

plt.figure(figsize=(5,5))
plt.tight_layout()
seabornInstance.distplot(dataset['TempHighF'])
plt.show()

#data splicing

x= dataset['TempLowF'].values.reshape(-1,1)
y= dataset['TempHighF'].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)  #training the algorithm

#To retrive the intercept:

print('intercept:',regressor.intercept_)

print('Coefficient:',regressor.coef_)   #beta values for every 1 unit change in the min temp the change in the max temp will be  0.9096

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':y_pred.flatten()})
print(df)

df1= df.head(25)
df1.plot(kind='bar',figsize=(10,10))
plt.grid(which='major',linestyle='-',linewidth='0.5',color='green')
plt.grid(which='minor',linestyle=':',linewidth='0.5',colour='red')
plt.show()

