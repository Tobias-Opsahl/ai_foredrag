import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def sigmoid(x):
    """ Returns the sigmoid function of x """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    """ Derivative of the sigmoid function """
    return sigmoid(x) * (1 - sigmoid(x))


def calculate_loss(y_hat, y):
    """ Loss (error) function of true value y and predicted value y_hat """
    mse = 1 / 2 * (y_hat - y) ** 2  # Mean squared error (MSE) / 2
    return mse


def d_loss(y, y_hat):
    """ Derivative of our loss function (MSE / 2) """
    return (y_hat - y)


def calculate_accuracy(preds, y):
    """
    Calculates the accuracy, given predictions and true values.

    Args:
        preds (list): List of floats. The predictions. For classification, use 0 and 1, not logits.
        y (list): The true values

    Returns:
        float: The accuracy.
    """
    accuracy = 0
    for i in range(len(preds)):
        accuracy += int(preds[i] == y[i])
    accuracy = accuracy * 100 / len(preds)
    return accuracy


def simple_neural_network(x1, x2, w1, w2, b):
    """
    Simple neural network with one layer and two input features.
    Outputs the predictions, given the inputs and the features.

    Args:
        x1 (float): First input
        x2 (float): Second input
        w1 (float): First weight
        w2 (float): Second weight
        b (float): Bias, konstantledd.

    Returns:
        float: The predictions.
    """
    z = w1 * x1 + w2 * x2 + b
    y_hat = sigmoid(z)
    return y_hat


def d_simple_neural_network(x1, x2, w1, w2, b, y):
    """
    Calculates the derivatives of the simple neural network.
    This is based on the mean-squared-error loss function.
    dL/dw = dL/dyhat * dyhat / dz * dz / dw

    Args:
        x1 (float): First input
        x2 (float): Second input
        w1 (float): First weight
        w2 (float): Second weight
        b (float): Bias, konstantledd.
        y (float): True prediction value

    Returns:
        d_w1, d_w2, d_b: The derivative of the weight1, weight2 and the bias.
    """
    y_hat = simple_neural_network(x1=x1, x2=x2, w1=w1, w2=w2, b=b)
    dL_dyhat = d_loss(y=y, y_hat=y_hat)
    dyhat_dz = y_hat * (1 - y_hat)
    dz_dw1 = x1
    dz_dw2 = x2
    dz_db = 1
    d_w1 = dL_dyhat * dyhat_dz * dz_dw1
    d_w2 = dL_dyhat * dyhat_dz * dz_dw2
    d_b = dL_dyhat * dyhat_dz * dz_db
    return d_w1, d_w2, d_b


def gradient_descent_step(x1, x2, w1, w2, b, y, learning_rate=0.1):
    """
    Performs one step of gradient descent.
    This means caclulating the derivatives of the parameters, and updating them with a learning rate.

    Args:
        x1 (float): First input
        x2 (float): Second input
        w1 (float): First weight
        w2 (float): Second weight
        b (float): Bias, konstantledd.
        y (float): True prediction value
        learning_rate (float, optional): The learning rate, determines how large step we take in the step.

    Returns:
        w1, w2, b: The updated parameters.
    """
    d_w1, d_w2, d_b = d_simple_neural_network(x1=x1, x2=x2, w1=w1, w2=w2, b=b, y=y)
    w1 = w1 - learning_rate * d_w1
    w2 = w2 - learning_rate * d_w2
    b = b - learning_rate * d_b
    return w1, w2, b


def perform_gradient_descent(x1_data, x2_data, y_data, w1_start, w2_start,
                             b_start, learning_rate=0.1, n_epochs=3, verbose=True):
    """
    Performs gradient descent for many datapoints and many epochs.
    This loops over the datapoints for every epoch, meaning we use batch-size 1.
    Naive and non-vectorized implementation.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1_start (float): Starting weight for weight 1
        w2_start (float): Starting weight for weight 2
        b_start (float): Starting value for bias.
        learning_rate (float, optional): Learning rate. Defaults to 0.1.
        n_epochs (int, optional): Amount of epochs to run for. Defaults to 3.
        verbose (bool, optional): If True, will print accuracy and loss after each epoch. Defaults to True.

    Returns:
        w1, w2, b: The weights after the gradient descent iterations.
    """
    w1 = w1_start
    w2 = w2_start
    b = b_start

    for epoch in range(n_epochs):
        for i in range(len(x1_data)):
            x1 = x1_data[i]
            x2 = x2_data[i]
            y = y_data[i]
            w1, w2, b = gradient_descent_step(x1=x1, x2=x2, w1=w1, w2=w2, b=b, y=y, learning_rate=learning_rate)

        y_hats = simple_neural_network(x1=x1_data, x2=x2_data, w1=w1, w2=w2, b=b)
        predictions = (y_hats > 0.5).astype(int)
        if verbose:
            accuracy = calculate_accuracy(predictions, y_data)
            loss = sum(calculate_loss(y_hats, y_data)) / len(x1_data)
            print(f"Accuracy after epoch {epoch}: {accuracy:.0f}%. ")
            print(f"Loss after epoch {epoch}: {loss:.4f}.\n")

    return w1, w2, b


def plot_predictions(x1_data, x2_data, y_data, w1, w2, b, ax=None, n_epochs=None, show=True):
    """
    Plots predictions done be network with parameters `w1`, `w2` and `b`.
    Plots the original datapoint, along with the decision boundary for the network.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1_start (float): Starting weight for weight 1
        w2_start (float): Starting weight for weight 2
        b_start (float): Starting value for bias.
        ax (axis, optional): The axis to plot on. Will make a new one if it is `None`.
        n_epochs (int, optional): If not None, will put the numbers of epochs used in the title.
        show (bool, optional): If True, shows the plot. If False, returns the axis. Defaults to True.

    Returns:
        ax: The axis plotted on.
    """
    if ax is None:
        _, ax = plt.subplots()

    # Create a mesh grid of points
    x1_min, x1_max = x1_data.min() - 1, x1_data.max() + 1
    x2_min, x2_max = x2_data.min() - 1, x2_data.max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))

    # Flatten the grid arrays and create a matrix where each row is a pair of points
    grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Vectorized prediction
    preds = simple_neural_network(grid[:, 0], grid[:, 1], w1, w2, b)
    preds = (preds > 0.5).astype(int)  # Convert to binary class labels (0 or 1)

    # Reshape the predictions to match the shape of the input grid
    preds = preds.reshape(xx1.shape)

    # Plot the decision regions by coloring the grid points
    ax.pcolormesh(xx1, xx2, preds, shading="auto", cmap="coolwarm_r", alpha=0.2)

    # Plot the data in two batches
    y_data = np.array(y_data)
    for color, marker, mask in zip(["blue", "red"], ["o", "x"], [y_data == 1, y_data != 1]):
        ax.scatter(x1_data[mask], x2_data[mask], color=color, marker=marker, alpha=0.7)

    y_hats = simple_neural_network(x1=x1_data, x2=x2_data, w1=w1, w2=w2, b=b)
    predictions = (y_hats > 0.5).astype(int)
    accuracy = calculate_accuracy(predictions, y_data)
    title = f"Accuracy: {accuracy:.0f}%"
    if n_epochs is not None:
        title += f" after {n_epochs} epochs"
    ax.set_title(title)
    ax.set_xlabel("Gjennomsnitts timer sollys")
    ax.set_ylabel("Gjennomsnitts temperatur")

    if show:
        plt.show()
    return ax  # return the Axes object for further manipulation outside the function


def read_data(filename="data.pkl"):
    """
    Reads the data made by `generate_data.py`.

    Args:
        filename (str, optional): Name of file to read. Defaults to "data.pkl".

    Returns:
        tuple: Tuple of the data.
    """
    data_dict = pickle.load(open(filename, "rb"))
    return data_dict["x1_data"], data_dict["x2_data"], data_dict["y_data"]


def get_float_input(prompt, default_value):
    """ Help function for maniging the input given by the user. """
    while True:
        user_input = input(prompt)
        if not user_input:
            return default_value
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def animate_gradient_descent(x1_data, x2_data, y_data, w1_start, w2_start, b_start, learning_rate=0.1, n_epochs=10,
                             epochs_per_frame=1):
    """
    Animates the gradient descent iterations.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1_start (float): Starting weight for weight 1
        w2_start (float): Starting weight for weight 2
        b_start (float): Starting value for bias.
        learning_rate (float, optional): Learning rate. Defaults to 0.1.
        n_epochs (int, optional): Amount of epochs to run for. Defaults to 3.
        interval (int, optional): Amount of milliseconds between each frame.
        verbose (bool, optional): If True, will print accuracy and loss after each epoch. Defaults to True.
    """
    fig, ax = plt.subplots()
    w1 = w1_start
    w2 = w2_start
    b = b_start

    # Run one update before starting the animation
    w1, w2, b = perform_gradient_descent(x1_data, x2_data, y_data, w1_start=w1, w2_start=w2,
                                         b_start=b, n_epochs=1, learning_rate=learning_rate)
    plot_predictions(x1_data=x1_data, x2_data=x2_data, y_data=y_data, w1=w1, w2=w2, b=b, ax=ax, show=False)

    def init():
        return ax,

    def update(frame):  # The form of the function that FuncAnimation wants
        nonlocal w1, w2, b  # Hacky way of overwriting the updated parameters for the next iteration
        ax.clear()  # clear previous frame
        w1, w2, b = perform_gradient_descent(x1_data, x2_data, y_data, w1_start=w1, w2_start=w2,
                                             b_start=b, n_epochs=epochs_per_frame, learning_rate=learning_rate)
        n_epoch = frame + 1
        plot_predictions(x1_data=x1_data, x2_data=x2_data, y_data=y_data, w1=w1, w2=w2, b=b, ax=ax, show=False,
                         n_epochs=n_epoch)
        return ax,

    ani = FuncAnimation(fig, update, init_func=init, frames=n_epochs, repeat=False, interval=100)
    plt.show()


if __name__ == "__main__":
    np.random.seed(57)
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--run", action="store_true", help="Optimize the neural network")

    args = parser.parse_args()
    w2 = 0.4  # Vekt for temperatur. Sann verdi 0.4
    x1_data, x2_data, y_data = read_data("data.pkl")

    if args.run:
        w1 = 0.1
        w2 = 1
        b = -10
        animate_gradient_descent(x1_data, x2_data, y_data, w1, w2, b, learning_rate=0.01, n_epochs=50)

    else:
        w1 = get_float_input("Skriv inn tall for vekt1, for timer med sollys, i intervallet [1, 5]: ", 0.5)
        b = get_float_input("Skriv inn tall for konstantledd, i intervallet [-20, -1]:", -10)
        plot_predictions(x1_data, x2_data, y_data, w1, w2, b)
