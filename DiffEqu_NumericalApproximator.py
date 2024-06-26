#######################################################################################################################
#                                                   James  Carroll                                                    #
#                                                      6/20/2024                                                      #
#######################################################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

""" 
    y' = 3(x^2)y - y
"""
def dydx(x, y):
    return (3 * (x**2) * y) - y

"""
    Initial conditions: Y at x = 1 is 3
    y(1) = 3
"""
x_0 = 1
y_0 = 3

"""
    Endpoint of numeric approximation:
    Find y at x = 1.5
    y(1.5) ~= ??.??
"""
x_n = 1.5

"""
    Solve for y(x) to compare approximation with exact result
    y(x) = 3 * exp(x^3 - x)
"""
def particular_solution(x):
    return 3 * math.exp((x**3) - x)

"""
    Parameters for the graph. 
    Moved here for easier configuration.
"""
graph_limits_x_min = 0.9
graph_limits_x_max = 1.6
graph_limits_y_min = 0
graph_limits_y_max = 25

graph_dydx_color = (0.1, 0.1, 0.75, 0.5)
graph_dydx_linewidth = 0.5

graph_eulers_color = "red"
graph_impEulers_color = "purple"
graph_rungeKutta_color = "orange"
graph_errorBar_color = "green"
graph_approx_linewidth = 1

graph_xy_resolution = 16 # Number of vectors to draw in x and y directions per unit change (0 to 1 along x axis contains 16 vectors)
graph_vector_length = 0.25

required_accuracy = 1e-2

#######################################################################################################################
#                         You should not have to mess with any of the code below this point.                          #
#######################################################################################################################

# known answer to compare approximations to
exact = particular_solution(x_n)

def eulers_method(x, y, h):
    k = dydx(x, y)
    return y + (h * k)

def improve_eulers_method(x, y, h):
    k1 = dydx(x, y)
    k2 = dydx(x + h, y + (h * k1))
    average_slope = (k1 + k2) / 2
    return y + (h * average_slope)

def runge_kutta_method(x, y, h,):
    k1 = dydx(x, y)
    k2 = dydx(x + (h / 2), y + ((h * k1) / 2))
    k3 = dydx(x + (h / 2), y + ((h * k2) / 2))
    k4 = dydx(x + h, y + (h * k3))
    return y + ((h / 6) * (k1 + (2 * k2) + (2 * k3) + k4))

# Make an approximation using the specified method and a number of steps
def numeric_approximate(n, approxFunc):
    xOut = [x_0] * (n + 1) # make a list to contain all x values used in the approximation
    yOut = [y_0] * (n + 1) # make a list to contain all y values used in the approximation
    x = x_0
    y = y_0
    h = (x_n - x_0) / n
    for t in range(n): # step through the approxiation, 0 to n - 1
        y = approxFunc(x, y, h) # approximate the y value at the next step
        x += h # calculate the x value at the next step
        xOut[t + 1] = x # save the next x value 
        yOut[t + 1] = y # save the next y value
    return xOut, yOut, y

# Approximate using n = 1 and test error. Increment if outside acceptable threshold, repeat.
#   xOut, yOut are the points used in the approximation. They are used to graph this approx.
def incremental_approach(approxFunc):
    xOut = [] # x part of approximation graph
    yOut = [] # y part of approximation graph
    n = 1
    error = 1
    result = 1
    print("\nApproximation using", approxFunc.__name__)
    while error >= required_accuracy:
        xOut, yOut, result = numeric_approximate(n, approxFunc)
        error = abs(exact - result)
        n += 1
    n -= 1 # subtract 1 from n because it is incremented before error is checked
    print("Error at step", n, "is %.4f" % error, "using", approxFunc.__name__)
    print("\tExact: %.4f" % exact, "Approximate: %0.4f" % result)
    return xOut, yOut, n

# perform numerical approximation on the function using all methods and display the minimum steps
print("Exact solution at y(%.2f)" % x_n, "= %.2f" % exact)

# Calculate the minimum number of steps required for each method
xPlt_eulers, yPlt_eulers, min_steps_euler = incremental_approach(eulers_method)
xPlt_impEulers, yPlt_impEulers, min_steps_imp_euler = incremental_approach(improve_eulers_method)
xPlt_rungeKutta, yPlt_rungeKutta, min_steps_runge_kutta = incremental_approach(runge_kutta_method)

# Print the results of the incremental_approach functions
print("\nNumber of steps required for Euler's method:", min_steps_euler, "\n")
print("Number of steps required for Improved Euler's method:", min_steps_imp_euler, "\n")
print("Number of steps required for Runge-Kutta method:", min_steps_runge_kutta, "\n")

#######################################################################################################################
#                              All the code below this point is for generating the graph.                             #
#######################################################################################################################

# Returns the highest number in the list
def max(numbers):
    max = numbers[0]
    for i in numbers:
        if i > max: max = i
    return max

# Returns the lowest number in the list
def min(numbers):
    min = numbers[0]
    for i in numbers:
        if i < min: min = i
    return min

def setup_graph():
    # Limits of y' graph
    xlims = [graph_limits_x_min, graph_limits_x_max]
    ylims = [graph_limits_y_min, graph_limits_y_max] # y(1.5) ~= 19.56

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.axvline(0, c="black")
    plt.axhline(0, c="black")

    sliderMax = max([min_steps_euler, min_steps_imp_euler, min_steps_runge_kutta])
    axSlider = plt.axes([0.25, 0.05, 0.65, 0.03])
    stepSlider = Slider(ax=axSlider, label='Number of steps', valmin=1, valmax=sliderMax, valinit=1, valstep=1)

    incAx = plt.axes([0.7, 0.025, 0.095, 0.03])
    decButton = Button(incAx, 'Decrement', color=graph_dydx_color, hovercolor='skyblue')
    decAx = plt.axes([0.8, 0.025, 0.095, 0.03])
    incButton = Button(decAx, 'Increment', color=graph_dydx_color, hovercolor='skyblue')

    return fig, axs, stepSlider, sliderMax, incButton, decButton # what are you doing step slider

# Generate the graph of y'
def generate_dydx_field():
    currXl, currXr = axs[0, 0].get_xlim()
    currYl, currYr = axs[0, 0].get_ylim()
    scl = ((currXr - currXl) / (graph_limits_x_max - graph_limits_x_min)) * graph_vector_length
    cRes = (currXr - currXl) * graph_xy_resolution
    mesh_width_x = (currXr - currXl) / cRes
    mesh_width_y = (currYr - currYl) / cRes
    dir_field_x_template = np.linspace((-mesh_width_x * scl) / cRes,
                                    (mesh_width_y * scl) / cRes, 100)
    for x in np.arange(currXl, currXr, mesh_width_x):
        for y in np.arange(currYl, currYr, mesh_width_y):
            curr_slope = dydx(x, y) # find the slope at the start of each vector
            curr_intercept = y - curr_slope * x 
            dir_field_xs = dir_field_x_template + x
            dir_field_ys = [curr_slope * dfx + curr_intercept for dfx in dir_field_xs]
            for i in range(2):
                for j in range(2):
                    axs[i, j].plot(dir_field_xs, dir_field_ys, color = graph_dydx_color, 
                        linewidth = graph_dydx_linewidth)

def clear_graph():
    currXl, currXr = axs[0, 0].get_xlim()
    currYl, currYr = axs[0, 0].get_ylim()
    for i in range(2):
                for j in range(2):
                    axs[i, j].clear()
    axs[0, 0].set_xlim([currXl, currXr])
    axs[0, 0].set_ylim([currYl, currYr])

def generate_graph(slider_value):
    generate_dydx_field()
    axs[0, 0].set_title("All approximation methods")

    # Generate the graph showing the approximation using euler's method
    numStepsEulers = min([min_steps_euler, slider_value])
    xPlt_eulers, yPlt_eulers, eulers_approx = numeric_approximate(numStepsEulers, eulers_method)
    axs[0, 0].plot(xPlt_eulers, yPlt_eulers, color = graph_eulers_color, linewidth = graph_approx_linewidth)
    axs[0, 1].plot(xPlt_eulers, yPlt_eulers, color = graph_eulers_color, linewidth = graph_approx_linewidth)
    axs[0, 1].set_title("Euler's Method using {0:.0f} steps = {1:.3f}".format(numStepsEulers, eulers_approx))

    # Generate the graph showing the approximation using improved euler's method
    numStepsImpEulers = min([min_steps_imp_euler, slider_value])
    xPlt_impEulers, yPlt_impEulers, impEulers_approx = numeric_approximate(numStepsImpEulers, improve_eulers_method)
    axs[0, 0].plot(xPlt_impEulers, yPlt_impEulers, color = graph_impEulers_color, linewidth = graph_approx_linewidth)
    axs[1, 0].plot(xPlt_impEulers, yPlt_impEulers, color = graph_impEulers_color, linewidth = graph_approx_linewidth)
    axs[1, 0].set_title("Improved Euler's Method {0:.0f} steps = {1:.3f}".format(numStepsImpEulers, impEulers_approx))

    # Generate the graph showing the approximation using runge-kutta method
    numStepsRungeKutta = min([min_steps_runge_kutta, slider_value])
    xPlt_rungeKutta, yPlt_rungeKutta, rungeKutta_approx = numeric_approximate(numStepsRungeKutta, runge_kutta_method)
    axs[0, 0].plot(xPlt_rungeKutta, yPlt_rungeKutta, color = graph_rungeKutta_color, linewidth = graph_approx_linewidth)
    axs[1, 1].plot(xPlt_rungeKutta, yPlt_rungeKutta, color = graph_rungeKutta_color, linewidth = graph_approx_linewidth)
    axs[1, 1].set_title("Runge-Kutta Method using {0:.0f} steps = {1:.3f}".format(numStepsRungeKutta, rungeKutta_approx))

    # Draw an error bar to represent the known value and acceptable range (1e-2)
    for i in range(2):
        for j in range(2):
            axs[i, j].errorbar(x_n, exact, yerr=0.01, fmt='o', color = graph_errorBar_color)

def update(val):
    clear_graph()
    generate_graph(val)

fig, axs, stepSlider, sliderMax, incButon, decButton = setup_graph()

def on_xlim_change(*args):
    clear_graph()
    generate_graph(stepSlider.val)

def increment_slider(event):
    if stepSlider.val >= sliderMax: # do not allow increment if at end of sliders range
        stepSlider.val = sliderMax
        return 
    stepSlider.set_val(stepSlider.val + 1)

def decrement_slider(event):
    if stepSlider.val <= 1: # do not allow decrement if at end of sliders range
        stepSlider.val = 1
        return 
    stepSlider.set_val(stepSlider.val - 1)

stepSlider.on_changed(update)
incButon.on_clicked(increment_slider)
decButton.on_clicked(decrement_slider)
fig.add_callback('xlim_changed',on_xlim_change)
generate_graph(stepSlider.val)
plt.show()