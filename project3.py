import numpy as np

from numpy.linalg import inv
import matplotlib.pyplot as plt

# Part 1
def gen_data():
    n = 100 # dimensions of data
    m = 1000 # number of data points
    X = np.random.normal(0,1, size=(m,n))
    a_true = np.random.normal(0,1, size=(n,1))
    y = X.dot(a_true) + np.random.normal(0,0.1,size=(m,1))

    return X, a_true, y


def objective_func(a_pred, X, y, reg_param=0.0):
    num_points = X.shape[0]
    y_pred = X.dot(a_pred)
    residual_sqr = (y_pred - y)**2

    # NOTE: Using MSE here instead of SE as in project handout
    return 0.5 * np.sum(residual_sqr, axis=0) + reg_param * np.linalg.norm(a_pred)**2


def closed_form_soln(X, y):
    """
    Solve for a using normal equations.
    """
    return np.dot(np.dot(inv(np.dot(X.T, X)), X.T), y)


def batch_gradient_descent(X, y, lrate, epsilon, num_iters, reg_param=0.0):
    m, n = X.shape
    a_curr = np.zeros(n)

    objective_func_values = []

    count = 0
    for _ in range(num_iters):
    #while True:
        residual = X.dot(a_curr) - y[:, 0]

        # Reshape for compatibility
        residual = residual.reshape(residual.shape[0], 1)
        prod = residual * X
        gradient = np.sum(residual * X, axis=0) + 2*reg_param*a_curr

        norm_grad = np.linalg.norm(gradient)
        if np.linalg.norm(gradient) < epsilon:
            break
        # Update a
        a_curr -= lrate * gradient

        obj_value = objective_func(a_curr, X, y[:, 0])
        #print "Iter: ", count
        #print "Objective: ", obj_value
        count += 1
        objective_func_values.append(obj_value)

    return a_curr, objective_func_values


def stochastic_gradient_descent(X, y, lrate, epsilon, num_iters=1000):
    m, n = X.shape
    a_curr = np.zeros(n)

    all_idx = xrange(m)
    objective_func_values = []

    #while True:
    for _ in range(num_iters):
        # Pick random point
        sample_idx = np.random.choice(all_idx, 1)
        sample = X[sample_idx, :].T[:, 0]

        residual = (sample.dot(a_curr) - y[sample_idx])[0,0]

        # if np.linalg.norm(residual) < epsilon:
        #     break
        # Update a
        a_curr -= lrate * residual * sample


        obj_value = objective_func(a_curr, X, y[:, 0])
        objective_func_values.append(obj_value)

    return a_curr, objective_func_values


# Part (c)/(d)
def gen_cd_data():
    train_m = 20
    test_m = 1000
    n = 100
    X_train = np.random.normal(0,1, size=(train_m,n))
    a_true = np.random.normal(0,1, size=(n,1))
    y_train = X_train.dot(a_true) + 0.5*np.random.normal(0,1,size=(train_m,1))
    X_test = np.random.normal(0,1, size=(test_m,n))
    y_test = X_test.dot(a_true) + 0.5*np.random.normal(0,1,size=(test_m,1))

    return X_train, y_train, X_test, y_test, a_true


if __name__ == "__main__":
    X, a_true, y = gen_data()
    #a_pred = closed_form_soln(X, y)

    epsilon = 0.001
    # (a).
    #print "Optimal a: ", a_pred
    #print "Objecive value: ", objective_func(a_pred, X, y)


    # (b).
    lrates = [0.0001, 0.001, 0.00125]
    # _, func_values_1 = batch_gradient_descent(X, y, lrates[0], epsilon, num_iters=20)
    # _, func_values_2 = batch_gradient_descent(X, y, lrates[1], epsilon, num_iters=20)
    # _, func_values_3 = batch_gradient_descent(X, y, lrates[2], epsilon, num_iters=20)
    #
    # print func_values_1
    # print func_values_2
    # print func_values_3
    #
    # # Plot objective func
    # num_iters = range(20)
    # fig, ax = plt.subplots()
    # ax.plot(num_iters, func_values_1, label="Lrate=0.0001")
    # ax.plot(num_iters, func_values_2, label="Lrate=0.001")
    # ax.plot(num_iters, func_values_3, label="Lrate=0.00125")
    # plt.title("Objective Value vs. Num Iterations")
    #
    # legend = ax.legend(loc='upper center', shadow=True)
    #
    # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # plt.show()


    # (c).
    lrates = [0.001, 0.01, 0.02]

    # _, func_values_1 = stochastic_gradient_descent(X, y, lrates[0], epsilon, num_iters=1000)
    # _, func_values_2 = stochastic_gradient_descent(X, y, lrates[1], epsilon, num_iters=1000)
    # _, func_values_3 = stochastic_gradient_descent(X, y, lrates[2], epsilon, num_iters=1000)
    #
    #
    # print func_values_1
    # print func_values_2
    # print func_values_3
    # # # Plot objective func
    # num_iters = range(1000)
    # fig, ax = plt.subplots()
    # ax.plot(num_iters, func_values_1, label="Lrate=0.001")
    # ax.plot(num_iters, func_values_2, label="Lrate=0.01")
    # ax.plot(num_iters, func_values_3, label="Lrate=0.02")
    # plt.title("Objective Value vs. Num Iterations")
    #
    # legend = ax.legend(loc='upper center', shadow=True)
    #
    # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')
    # plt.show()

    # (d).
    lrate = 0.00125
    X_train, y_train, X_test, y_test, a_true = gen_cd_data()
    # a_pred, obj_func_values = batch_gradient_descent(X_train, y_train, lrate, 0.01, 1000)
    # print "Test objective: ", objective_func(a_pred, X_test, y_test[:, 0])


    # (e).
    reg_params = [10.0**x for x in range(2, -4, -1)]
    for reg in reg_params:
        X_train, y_train, X_test, y_test, a_true = gen_cd_data()
        a_pred, obj_func_values = batch_gradient_descent(X_train, y_train, lrate, 0.01, 1000, reg_param=reg)
        print "Reg param: ", reg
        print "Train objective: ", obj_func_values[-1]
        print "Test objective: ", objective_func(a_pred, X_test, y_test[:, 0])
