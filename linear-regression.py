import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import pickle
from matplotlib import pyplot as plt




#height=[[4.0],[4.5],[5.0],[5.2],[5.4],[5.8],[6.1],[6.2],[6.4],[6.8]]
#weight=[  42 ,  44 , 49, 55  , 53  , 58   , 60  , 64  ,  66 ,  69]

# print("height weight")
# for row in zip(height, weight):
#     print(row[0][0],"->",row[1])

# plt.scatter(height,weight,color='black')
# plt.xlabel("height")
# plt.ylabel("weight")
# plt.show()

#run the prep package and get the data frame
# Use the Azure Machine Learning data preparation package
from azureml.dataprep import package

# This call will load the referenced package and return a DataFrame.
# If run in a PySpark environment, this call returns a
# Spark DataFrame. If not, it will return a Pandas DataFrame.
df = package.run('Prep4.dprep', dataflow_idx=0,spark=False)

# Remove this line and add code that uses the DataFrame
print(df.head(10))
height = pd.DataFrame(df, columns=["Heightft"]).values
weight = pd.DataFrame(df, columns=["Weightkg"]).values

plt.scatter(height,weight,color='black')
plt.xlabel("height")
plt.ylabel("weight")
plt.show()

reg=linear_model.LinearRegression()
reg.fit(height,weight)

#emit slope and intercept
m=reg.coef_[0]
b=reg.intercept_
print("slope=",m, "intercept=",b)

#use slope and intercept to fit our data points
# plt.scatter(height,weight,color='black')
# predicted_values = [reg.coef_ * i + reg.intercept_ for i in height]
# plt.plot(height, predicted_values, 'b')
# plt.xlabel("height")
# plt.ylabel("weight")
# plt.show()


predictedVal = reg.predict(6.1)
print(predictedVal)


print ("Export the model to model.pkl")
f = open('./outputs/model.pkl', 'wb')
pickle.dump(reg, f)
f.close()
