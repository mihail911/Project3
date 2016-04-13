import numpy as np

from numpy.linalg import inv


# Part 1
def gen_data():
    n = 100 # dimensions of data
    m = 1000 # number of data points
    X = np.random.normal(0,1, size=(m,n))
    a_true = np.random.normal(0,1, size=(n,1))
    y = X.dot(a_true) + np.random.normal(0,0.1,size=(m,1))

    return X, a_true, y


def objective_func(a_pred, X, y):
    num_points = X.shape[0]
    y_pred = X.dot(a_pred)
    residual_sqr = (y_pred - y)**2

    return 0.5 * np.sum(residual_sqr, axis=0) / num_points


def closed_form_soln(X, y):
    """
    Solve for a using normal equations.
    """
    return np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)


def batch_gradient_descent(X, y, lrate, epsilon):
    m, n = X.shape
    a_curr = np.zeros(n)

    all_idx = xrange(m)


    while True:
        residual = X.dot(a_curr) - y

        # Reshape for compatibility
        residual = residual.reshape(residual.shape[0], 1)
        gradient = 2/m * np.sum(residual * X, axis=0)
        if np.linalg.norm(gradient) < epsilon:
            break
        # Update a
        a_curr -= lrate * gradient


    return a_curr


def stochastic_gradient_descent(X, y, lrate, epsilon):
    m, n = X.shape
    a_curr = np.zeros(n)

    all_idx = xrange(m)

    while True:
        # Pick random point
        sample_idx = np.random.choice(all_idx, 1)
        sample = X[sample_idx, :]

        residual = sample.dot(a_curr) - y[sample_idx]

        if np.linalg.norm(residual) < epsilon:
            break
        # Update a
        a_curr -= lrate * residual * sample


    return a_curr


if __name__ == "__main__":
    X, a_true, y = gen_data()
    a_pred = closed_form_soln(X, y)

    # (a).
    print "Optimal a: ", a_pred
    print "Objecive value: ", objective_func(a_pred, X, y)


    # (b).
    lrates = [ 0.0001, 0.001, 0.00125]


    # (c).
    lrates = [0.001, 0.01, 0.02]