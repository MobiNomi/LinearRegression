import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
data = pd.read_csv("C:\\Users\\Mubashir Hussain\\3D Objects\\Mlmodel\\pythonProject\\Salary_Data.csv")

#checking data
# print(data.head())
# print(data.tail())

# print(data.describe())

# print(data.info())

#null values

# print(data.isnull().sum())

# duplicate values 
# print(data.duplicated().sum())


# looking for the otliers in data 

df = pd.DataFrame(data)
# print(df.head())

# sns.boxplot(data=df["Salary"])
# plt.show()

#distribution of data
# sns.kdeplot(df["Salary"], fill=True)

# sns.boxplot(data=df["Salary"])
# plt.show()



q1 = df["Salary"].quantile(0.25)
q2 = df["Salary"].quantile(0.50)
q3 = df["Salary"].quantile(0.75)

print(q1)
print(q3)

IQR = q3 - q1

print(IQR)
lower = q1 - 1.5*IQR
upper = q3 + 1.5*IQR
filtered_df = df[(df["Salary"] >= lower) & (df["Salary"] <= upper)]

# print(filtered_df.count())




# scalar = MinMaxScaler()
# filtered_df[["YearsExperience" ,"Salary"]] = scalar.fit_transform(filtered_df[["YearsExperience" ,"Salary"]])

# print(df.head())


# now applying the linear regression model on the data 


X = filtered_df.iloc[: , : -1].values #input for regression model

y = filtered_df.iloc[: , -1].values #last column that is output for the model 

x_train ,x_test, y_train ,y_test = train_test_split(X,y,test_size=1/3,random_state=0) #for training and testing the model

# print(x_test.count()) # checking the number of obervations for the testing purpose

model =  LinearRegression()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)



comparison = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print(comparison.head(10))  # Show first 10 rows


plt.scatter(x_train[:, 0], y_train, color='red')
plt.plot(x_train[:, 0], model.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(x_test[:, 0], y_test, color='red')
plt.plot(x_train[:, 0], model.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()