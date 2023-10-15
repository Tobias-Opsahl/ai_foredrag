import pickle
import numpy as np
from simple_nn import simple_neural_network


def generate_data_simple(x1_data, x2_data, w1, w2, b, sd=0, filename="data.pkl"):
    """
    Generates data according to a simple_nerual_networks predictions, given its weights.
    May also add noise with `sd`.

    Args:
        x1_data (list): List of x1 datapoints. The y_data will be made accordingly for this and x2.
        x2_data (list): List of x2 datapoints.
        w1 (float): Weight one of the network
        w2 (float): Weight two of the network.
        b (float): The bias of the network
        sd (float, optional): The standard deviation of gaussian noise added to the points.
            0 means no noise. Defaults to 0.
        filename (str, optional): Filename for the data to be saved. Remember to add `.pkl`.
            Defaults to "data.pkl".
    """
    generated_data = []
    for i in range(len(x1_data)):
        x1 = x1_data[i]
        x2 = x2_data[i]
        y_hat = simple_neural_network(x1=x1, x2=x2, w1=w1, w2=w2, b=b)
        y_new = y_hat + np.random.normal(0, sd, 1)[0]
        y = int(y_new > 0.5)
        generated_data.append(y)
    data_dict = {"x1_data": x1_data, "x2_data": x2_data, "y_data": generated_data}
    with open(filename, "wb") as outfile:
        pickle.dump(data_dict, outfile)


def quadratic_function(x, a=0.4, b=-1.3, c=1.5):
    """
    Simple quadratic function. Used to generate data.
    Given a point and coefficient, returns the output y.

    Args:
        x (float): The input.
        a (float, optional): The quadratic coefficient. Defaults to 0.4.
        b (float, optional): The linear coefficient. Defaults to -1.3.
        c (float, optional): The constant term. Defaults to 1.5.

    Returns:
        float: The output of the equation.
    """
    return a * x ** 2 + b * x + c


def generate_data_polynomial(x1_data, x2_data, polynomial_function, filename="polynomial_data.pkl"):
    """
    Generates binary data with two input features, according to some polynomial function.

    Args:
        x1_data (list): List of x1-datapoints.
        x2_data (list): List of x2-datapoints
        polynomial_function (callable): Function that separates data and determines `y_data`
        filename (str, optional): Filename of the saved data. Remember to add `.pkl`. Defaults to "poynomial_data.pkl".
    """
    y_data = []
    for i in range(len(x1_data)):
        x1 = x1_data[i]
        x2 = x2_data[i]
        y = 1 if x2 > polynomial_function(x1) else 0
        y_data.append(y)
    data_dict = {"x1_data": x1_data, "x2_data": x2_data, "y_data": y_data}
    with open(filename, "wb") as outfile:
        pickle.dump(data_dict, outfile)


if __name__ == "__main__":
    np.random.seed(57)

    # Generate simple-nn data
    n_data_points = 100
    x1_data = np.random.uniform(0, 12, n_data_points)
    x2_data = np.random.uniform(0, 40, n_data_points)
    w1 = round(np.sqrt(223) ** 1.3 / 10 * 0.69, 1)  # "encrypted" value of step one of the exercises
    w2 = 0.4
    b = -int(np.sqrt(193) ** 1.3 * (1 / 1.4 ** 2) + 2)  # "encrypted" value for step one of exercises
    generate_data_simple(x1_data, x2_data, w1, w2, b)

    # Generate polynomial data
    np.random.seed(57)
    n_data_points = 500
    x1_data = np.linspace(-5, 5, n_data_points)
    x2_data = np.random.uniform(-5, 5, n_data_points)
    generate_data_polynomial(x1_data, x2_data, quadratic_function)
