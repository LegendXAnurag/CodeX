import numpy
import randomdata
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Selecting the number of sample data
#always put numPoint>1 else undesired output will be obtained
numPoint=10
#setting actual mass
trueMass=5
#Generating datapoints
acceleration, force=randomdata.Dataset(trueMass,numPoint)
#Plotting the given data points
plt.scatter(acceleration,force,color="blue",label="Data Points", marker="o",s=20)
plt.title("Force vs Acceleration")
plt.xlabel("Acceleration (m/s^2)")
plt.ylabel("Force (N)")
#Reshaping the 1D array of accn to 2D array so that it can fit inside Linear Regression Model
accelerationReshaped=acceleration.reshape(-1,1)
model = LinearRegression()      #Creating Linear Regression model
model.fit(accelerationReshaped,force)
mass=model.coef_[0]     #Extracting Estimated Mass
print("Estimated Mass:",mass)
diff=mass-trueMass
if(diff>=0):
    error=diff/trueMass*100
else:
    error=-diff/trueMass*100
print("Percentage Error:",error,"%")
predictedForce=model.predict(accelerationReshaped)  #Creating array of corresponding forces based on predicted mass
#Plotting the best fit line
plt.plot(acceleration,predictedForce,color="r", label="Best Fit Line")
plt.legend()    #Adding legend, grid
plt.grid()
plt.show()
