# DiffEqu_NumericalApproximator
Python script for visualizing various forms of numerical approximation of the solutions to ordinary differential equations.
This program was designed to answer a question for a differential equations exam that asked for the minimum number of steps required so that the approximate value is within 1e-2 of the known answer. 
![Example solution](/Figure_1.png)

## Usage
- Replace the 'dydx' function with the equation for y' for which you want to approximate the solution of.
- Replace the 'particular_solution' function with the known solution for y(x), if one exists.
- Replace the x_0, y_0, and x_n variables with the inital conditions and endpoint for the approximation.
- Run the program. It will automatically determine the minimum number of steps to make the approximation.
- After computing the minimum steps for the required accuracy, the program will generate a graph to help visualize what is going on.
