import numpy
import randomdata
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
#Selecting the number of sample data
#always put numPoint>1 else undesired output will be obtained
numPoint=10
#setting actual mass
trueMass=5
#Generating datapoints
acceleration, force=randomdata.Dataset(trueMass,numPoint)
aTrain,aTest,fTrain,fTest=train_test_split(acceleration,force,random_state=20,test_size=0.25)
#Plotting the given data points
plt.scatter(aTrain,fTrain,color="blue",label="Data Points", marker="o",s=20)
plt.title("Force vs Acceleration")
plt.xlabel("Acceleration (m/s^2)")
plt.ylabel("Force (N)")
#Reshaping the 1D array of accn to 2D array so that it can fit inside Linear Regression Model
aTrainReshape=aTrain.reshape(-1,1)
model = LinearRegression()      #Creating Linear Regression model
model.fit(aTrainReshape,fTrain)
mass=model.coef_[0]     #Extracting Estimated Mass
print("Estimated Mass:",mass)
predictedForce=model.predict(aTest.reshape(-1,1))  #Creating array of corresponding forces based on predicted mass
#Plotting the best fit line
from sklearn import metrics
print(metrics.mean_absolute_error(fTest,predictedForce)) 
plt.plot(aTest,predictedForce,color="r", label="Best Fit Line")
plt.legend()    #Adding legend, grid
plt.grid()
plt.show()
