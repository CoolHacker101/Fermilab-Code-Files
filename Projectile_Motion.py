from math import pi
from numpy import cos
cos(pi/4)

# (1) Import matplotlib so that we can make plots.  :)  
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('default')
plt.rcParams['figure.figsize'] = [7,5]

# (2) This example stores our x and y data points in numpy arrays.  
#     You can also put your x and y data points in a python list object.
import numpy as np

# (3) Create an array or list of x values
dx = 0.5
xmax = 10
x = np.arange(0, xmax+dx, dx) #array from 0 to xmax in steps dx

# (4) Create an array or list of y values
y1 = x     
y2 = x**2
y3 = x**3

# (5) Make a Matplotlib scatter plot
plt.scatter(x, x, label='linear')  #the first argument of plt.scatter is the list/array of x values
plt.scatter(x, x**2, label='quadratic') #The second argument of plt.scatter is the list/array of y values
plt.scatter(x, x**3, label='cubic') #You can optionally give each data set its own 'label', which is shown in the legend

plt.title("A scatter plot created with matplotlib.pyplot.scatter()") #title
plt.xlabel('x') #x-axis label
plt.ylabel('y(x)')  #y-axis label
plt.legend() #the legend won't show up without this

# (6) Show the plot in the display window.
plt.show()
plt.clf()

#Constants
g = 9.80665 #m/^2, CODATA 2018 value (https://physics.nist.gov/cgi-bin/cuu/Value?gn)
dt= 0.1 #step size in seconds
m = 10  #kg

#Initial position
y0 = 0  #m
x0 = 0  #m
t0 = 0  #s
r0 = np.array([x0,y0,0]) #Store initial-position vector as 3-element numpy arrays 

#Initial velocity
speed = 20 #m/s
theta_deg = 45.0 #degrees
theta = np.pi*theta_deg/180 
vx = speed*np.cos(theta)
vy = speed*np.sin(theta)
v0 = np.array([vx,vy,0]) #Store initial-velocity vector as 3-element numpy arrays 

# ==> Calculate the initial momentum and net force 
p0 = m*v0
Fnet = np.array([0, -m*g, 0]) #only gravity present

# ==> Initiate the loop variables here <==
t = t0
r = r0
p = p0

# ==> Simulate the ball's motion by creating a while loop that stops when the ball hits the ground

X_Values = []
Y_Values = []
T_Values = []

while (r[1] >= 0):
    p = p + Fnet*dt             #update the momentum using the net force
    v = p/m                     #calculate the new velocity
    r = r + v*dt                #update the position using the velocity
    t = t+dt                    #increment the simulation time
    X_Values.append(r [0])
    Y_Values.append(r [1])
    T_Values.append(t)
    
print(r [0] )
print(t)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1) 
ax1.scatter(T_Values, X_Values)
ax2.scatter(T_Values, Y_Values)
ax3.scatter(X_Values, Y_Values)
plt.show()

print("Predictions from kinematics equations:")
    print(f"-- range={xf} m\%.")
    print(f"-- landing time t={tf} s")
    
    def function_T(v0, theta):
    t = 2*v0*np.sin(theta)/g
    return t
def function_R(x0, v0, theta, t):
    r = v0*np.cos(theta)*t-x0
    return r

Landing_Time = function_T(speed, theta)
Range = function_R(x0, speed, theta, Landing_Time)

