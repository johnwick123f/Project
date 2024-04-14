import sympy as sp
import numpy as np
from PIL import Image, ImageDraw

def plot_equation_pil(equation, size=(400, 400), xlim=None, ylim=None):
    """
    Plot the graph of a given equation and return it as a PIL image.

    Parameters:
    - equation: str, the equation in the form 'y = f(x)'.
    - size: tuple, size of the output image (default is (400, 400)).
    - xlim: tuple, limits for the x-axis (default is None).
    - ylim: tuple, limits for the y-axis (default is None).

    Returns:
    - PIL Image
    """

    # Parse the equation
    x = sp.symbols('x')
    y = sp.sympify(equation.replace('^', '**').replace('x', '*x').split('=')[1].strip())

    # Convert the equation to a lambda function for numerical evaluation
    func = sp.lambdify(x, y, modules=['numpy'])

    # Generate x values
    x_values = np.linspace(-10, 10, size[0])

    # Calculate y values
    y_values = func(x_values)

    # Normalize y values to fit within the image height
    min_y, max_y = min(y_values), max(y_values)
    y_values = (y_values - min_y) / (max_y - min_y) * size[1]

    # Create a blank white image
    image = Image.new('RGB', size, 'white')
    draw = ImageDraw.Draw(image)

    # Plot the graph
    for i in range(len(x_values) - 1):
        draw.line([(i, size[1] - int(y_values[i])), ((i + 1), size[1] - int(y_values[i + 1]))], fill='black')

    # Add labels and title
    draw.text((size[0] // 2, size[1] - 20), 'x', fill='black')
    draw.text((size[0] - 20, size[1] // 2), 'y', fill='black')
    draw.text((10, 10), f'Plot of {equation}', fill='black')

    # Return the image
    return image
