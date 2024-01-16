from simple_nn import read_data, animate_gradient_descent


if __name__ == "__main__":
    x1_data, x2_data, y_data = read_data("data.pkl")
    w1 = 0.1
    w2 = 1
    b = -10
    animate_gradient_descent(x1_data, x2_data, y_data, w1, w2, b, learning_rate=0.01, n_epochs=50)
