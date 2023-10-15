import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from simple_nn import read_data, sigmoid, calculate_accuracy, calculate_loss, get_float_input


def two_layer_neural_network(x1, x2, w1, w2, w3, w4, w5, w6, b1, b2):
    """
    A simple neural network with one hidden layer that uses the sigmoid activation function.

    Args:
        x1 (float): First input feature.
        x2 (float): Second input feature.
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.

    Returns:
        float: The output of the network.
    """
    # Hidden layer
    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w3 * x1 + w4 * x2 + b1
    a2 = sigmoid(z2)

    # Output layer
    z3 = w5 * a1 + w6 * a2 + b2
    output = sigmoid(z3)

    return output


def derivatives_mse(x1, x2, y, w1, w2, w3, w4, w5, w6, b1, b2):
    """
    Calculate the derivatives of the loss with respect to each weight and bias, assuming an MSE loss function.

    Args:
        x1, x2 (float): Input features.
        y (float): True label.
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.

    Returns:
        tuple: A tuple containing derivatives with respect to each weight and bias.
    """
    # Forward pass
    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)
    z2 = w3 * x1 + w4 * x2 + b1
    a2 = sigmoid(z2)
    z3 = w5 * a1 + w6 * a2 + b2
    output = sigmoid(z3)

    # Backward pass
    error = output - y

    # Derivatives for the output layer
    output_deriv = error * output * (1 - output)

    d_w5 = output_deriv * a1
    d_w6 = output_deriv * a2
    d_b2 = output_deriv

    # Derivatives for the hidden layer
    hidden_deriv_1 = w5 * output_deriv * a1 * (1 - a1)
    hidden_deriv_2 = w6 * output_deriv * a2 * (1 - a2)

    d_w1 = hidden_deriv_1 * x1
    d_w2 = hidden_deriv_1 * x2
    d_w3 = hidden_deriv_2 * x1
    d_w4 = hidden_deriv_2 * x2
    d_b1 = hidden_deriv_1 + hidden_deriv_2  # since both neurons in the hidden layer share the same bias

    return d_w1, d_w2, d_w3, d_w4, d_w5, d_w6, d_b1, d_b2


def derivatives_bce(x1, x2, y, w1, w2, w3, w4, w5, w6, b1, b2):
    """
    Perform one step of gradient descent on the neural network with a hidden layer using the specified derivatives.

    Args:
        x1 (float): First input feature.
        x2 (float): Second input feature.
        y (float): True label.
        w1 (float): Weight of the network.
        w2 (float): Weight of the network.
        w3 (float): Weight of the network.
        w4 (float): Weight of the network.
        w5 (float): Weight of the network.
        w6 (float): Weight of the network.
        b1 (float): Bias of the network.
        b2 (float): Bias of the network.
        learning_rate (float, optional): The learning rate. Defaults to 0.1.
        derivatives (function, optional): Function for calculating derivatives, either `derivatives_mse` or
            `derivatives_bce`. If None, `derivatives_bce` is used. Defaults to None.

    Returns:
        tuple: Updated weights and biases.
    """
    # Forward pass
    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)
    z2 = w3 * x1 + w4 * x2 + b1
    a2 = sigmoid(z2)
    z3 = w5 * a1 + w6 * a2 + b2
    output = sigmoid(z3)

    # Simple derivative for BCE
    error = output - y

    # Derivatives for the output layer
    d_w5 = error * a1
    d_w6 = error * a2
    d_b2 = error

    # Derivatives for the hidden layer
    hidden_deriv_1 = w5 * error * a1 * (1 - a1)
    hidden_deriv_2 = w6 * error * a2 * (1 - a2)

    d_w1 = hidden_deriv_1 * x1
    d_w2 = hidden_deriv_1 * x2
    d_w3 = hidden_deriv_2 * x1
    d_w4 = hidden_deriv_2 * x2
    d_b1 = hidden_deriv_1 + hidden_deriv_2  # since both neurons in the hidden layer share the same bias

    return d_w1, d_w2, d_w3, d_w4, d_w5, d_w6, d_b1, d_b2


def gradient_descent_step(x1, x2, y, w1, w2, w3, w4, w5, w6, b1, b2, learning_rate=0.1, derivatives=None):
    """
    Calculate the derivatives of the BCE loss with respect to each weight and bias.

    Args:
        x1, x2 (float): Input features.
        y (float): True label.
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.

    Returns:
        tuple: A tuple containing derivatives with respect to each weight and bias.
    """
    if derivatives is None:
        derivatives = derivatives_bce
    d_w1, d_w2, d_w3, d_w4, d_w5, d_w6, d_b1, d_b2 = derivatives(
        x1, x2, y, w1, w2, w3, w4, w5, w6, b1, b2)

    # Update the weights and biases
    w1 -= learning_rate * d_w1
    w2 -= learning_rate * d_w2
    w3 -= learning_rate * d_w3
    w4 -= learning_rate * d_w4
    w5 -= learning_rate * d_w5
    w6 -= learning_rate * d_w6
    b1 -= learning_rate * d_b1
    b2 -= learning_rate * d_b2

    return w1, w2, w3, w4, w5, w6, b1, b2


def perform_gradient_descent(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2,
                             learning_rate=0.1, n_epochs=3, verbose=True, derivatives=None):
    """
    Performs gradient descent for many datapoints and many epochs.
    This loops over the datapoints for every epoch, meaning we use batch-size 1.
    Naive and non-vectorized implementation.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.
        learning_rate (float, optional): Learning rate. Defaults to 0.1.
        n_epochs (int, optional): Amount of epochs to run for. Defaults to 3.
        verbose (bool, optional): If True, will print accuracy and loss after each epoch. Defaults to True.

    Returns:
        w1, w2, b: The weights after the gradient descent iterations.
    """
    for epoch in range(n_epochs):
        for i in range(len(x1_data)):
            x1 = x1_data[i]
            x2 = x2_data[i]
            y = y_data[i]
            w1, w2, w3, w4, w5, w6, b1, b2 = gradient_descent_step(x1, x2, y, w1, w2, w3, w4, w5, w6, b1, b2,
                                                                   learning_rate=learning_rate, derivatives=derivatives)

        y_hats = two_layer_neural_network(x1_data, x2_data, w1, w2, w3, w4, w5, w6, b1, b2)
        predictions = (y_hats > 0.5).astype(int)
        if verbose:
            accuracy = calculate_accuracy(predictions, y_data)
            loss = sum(calculate_loss(y_hats, y_data)) / len(x1_data)
            print(f"Accuracy after epoch {epoch}: {accuracy:.0f}%. ")
            print(f"Loss after epoch {epoch}: {loss:.4f}.\n")

    return w1, w2, w3, w4, w5, w6, b1, b2


def plot_predictions(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2, ax=None, n_epochs=None, show=True):
    """
    Plots predictions done be network with parameters `w1`, `w2` and `b`.
    Plots the original datapoint, along with the decision boundary for the network.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.
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
    preds = two_layer_neural_network(grid[:, 0], grid[:, 1], w1, w2, w3, w4, w5, w6, b1, b2)
    preds = (preds > 0.5).astype(int)  # Convert to binary class labels (0 or 1)

    # Reshape the predictions to match the shape of the input grid
    preds = preds.reshape(xx1.shape)

    # Plot the decision regions by coloring the grid points
    ax.pcolormesh(xx1, xx2, preds, shading="auto", cmap="coolwarm_r", alpha=0.2)

    # Plot the data in two batches
    y_data = np.array(y_data)
    for color, marker, mask in zip(["blue", "red"], ["o", "x"], [y_data == 1, y_data != 1]):
        ax.scatter(x1_data[mask], x2_data[mask], color=color, marker=marker, alpha=0.7)

    y_hats = two_layer_neural_network(x1_data, x2_data, w1, w2, w3, w4, w5, w6, b1, b2)
    predictions = (y_hats > 0.5).astype(int)
    accuracy = calculate_accuracy(predictions, y_data)
    title = f"Accuracy: {accuracy:.0f}%"
    if n_epochs is not None:
        title += f" after {n_epochs} epochs"
    ax.set_title(title)
    ax.set_xlabel("x1-data")
    ax.set_ylabel("x2-data")

    if show:
        plt.show()
    return ax  # return the Axes object for further manipulation outside the function


def animate_gradient_descent(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2, learning_rate=0.1, n_epochs=10,
                             epochs_per_frame=1):
    """
    Animates the gradient descent iterations.

    Args:
        x1_data (list): List of x1-datapoints
        x2_data (list): List of x2-datapoints
        y_data (list): List of y-data points (true values).
        w1, w2, w3, w4, w5, w6 (float): Weights of the network.
        b1, b2 (float): Biases of the network.
        learning_rate (float, optional): Learning rate. Defaults to 0.1.
        n_epochs (int, optional): Amount of epochs to run for. Defaults to 3.
        interval (int, optional): Amount of milliseconds between each frame.
        verbose (bool, optional): If True, will print accuracy and loss after each epoch. Defaults to True.
    """
    fig, ax = plt.subplots()

    # Run one update before starting the animation
    w1, w2, w3, w4, w5, w6, b1, b2 = perform_gradient_descent(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2,
                                                              n_epochs=1, learning_rate=learning_rate)

    def init():
        return ax,

    def update(frame):  # The form of the function that FuncAnimation wants
        nonlocal w1, w2, w3, w4, w5, w6, b1, b2  # Hacky way of overwriting the updated parameters for the next iter
        ax.clear()  # clear previous frame
        w1, w2, w3, w4, w5, w6, b1, b2 = perform_gradient_descent(
            x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2, n_epochs=epochs_per_frame,
            learning_rate=learning_rate)
        n_epoch = frame + 1
        plot_predictions(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2, ax=ax, show=False,
                         n_epochs=n_epoch)
        return ax,

    anim = FuncAnimation(fig, update, init_func=init, frames=n_epochs, repeat=False, interval=10)
    return anim


if __name__ == "__main__":
    np.random.seed(57)
    x1_data, x2_data, y_data = read_data("polynomial_data.pkl")

    w1 = 0.001
    w2 = -0.001
    w3 = -0.001
    w4 = -0.001
    w5 = 0.001
    w6 = -0.001
    b1 = -0.001
    b2 = 0.001

    message = "Oppgi learning-rate for nettverket, på intervallet [0.001, 1]. Trykk enter for default verdri 0.01: "
    learning_rate = get_float_input(message, 0.01)
    anim = animate_gradient_descent(x1_data, x2_data, y_data, w1, w2, w3, w4, w5, w6, b1, b2,
                                    learning_rate=learning_rate, n_epochs=100, epochs_per_frame=1)
    print(w1, w2, w3, w4, w5, w6, b1, b2)
    plt.show()
