import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

def plot_equation(equation, xlim=None, ylim=None):
    """
    Plot the graph of a given equation.

    Parameters:
    - equation: str, the equation in the form 'y = f(x)'.
    - xlim: tuple, limits for the x-axis (default is None).
    - ylim: tuple, limits for the y-axis (default is None).

    Returns:
    - None
    """

    # Parse the equation
    x = sp.symbols('x')
    y = sp.sympify(equation.replace('^', '**').replace('x', '*x').split('=')[1].strip())

    # Convert the equation to a lambda function for numerical evaluation
    func = sp.lambdify(x, y, modules=['numpy'])

    # Generate x values
    x_values = np.linspace(-10, 10, 400)

    # Calculate y values
    y_values = func(x_values)

    # Plot the graph
    plt.plot(x_values, y_values)

    # Add labels and title
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Plot of {equation}')

    # Set axis limits if specified
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Show the plot
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.show()
