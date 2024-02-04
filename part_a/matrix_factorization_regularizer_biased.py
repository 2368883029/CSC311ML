from utils import *
from scipy.linalg import sqrtm
from matrix_factorization_regularizer import accs_l2, val_l2_sqres, train_l2_sqres, als_L2constraint

import numpy as np
import matplotlib.pyplot as plt

counter_l2 = 0
accs_bias = []
train_bias_sqres = []
val_bias_sqres = []

def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss

def update_u_z_bias(train_data, lr, u, z, b_u, b_z, lbd, lbd_b):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    b_u_col = b_u.reshape(-1, 1)
    error = c - (np.dot(u[n], z[q]) + b_u[n] + b_z[q])

    # Update U and Z based on the error
    u[n] += lr * (error * z[q] - lbd * u[n])
    z[q] += lr * (error * u[n] - lbd * z[q])
    b_u[n] += lr * (error - lbd_b * b_u[n])
    b_z[q] += lr * (error - lbd_b * b_z[q])

    ####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z, b_u, b_z


def als_bias(train_data, k, lr, num_iteration, lbd, lbd_b, val_data=None, len_user=None, len_question=None):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :param lbd: float
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}, if want to record MSE
    :param len_user: int, length of user if use bootstrapping
    :param len_question: int, length of question if use bootstrapping
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    if not len_user or not len_question:
        len_user = len(set(train_data["user_id"]))
        len_question = len(set(train_data["question_id"]))

    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len_user, k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len_question, k))
    b_u = np.zeros(len_user)
    b_z = np.zeros(len_question)

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z, b_u, b_z = update_u_z_bias(train_data, lr, u, z, b_u, b_z, lbd, lbd_b)
        if i % 10000 == 0 and val_data:
            b_u_col = b_u.reshape(-1, 1)
            mat = u @ z.T + b_u_col + b_z
            acc = sparse_matrix_evaluate(val_data, mat)
            train_sqre = squared_error_loss(train_data, u, z)
            val_sqre = squared_error_loss(val_data, u, z)
            accs_bias.append(acc)
            train_bias_sqres.append(train_sqre)
            val_bias_sqres.append(val_sqre)

    b_u_col = b_u.reshape(-1, 1)
    mat = u @ z.T + b_u_col + b_z
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # Part d
    reconstruct_matrix_bias = als_bias(train_data, 50, 0.05, 250000, 0.025, 0.025, val_data=val_data)
    reconstruct_matrix_l2 = als_L2constraint(train_data, 50, 0.05, 250000, 0.025, val_data=val_data)

    # PART e
    iterations_lst = list(range(0, 250000, 10000))

    plt.figure()
    plt.plot(iterations_lst, accs_bias, label="Accuracy of Algorithm with bias")
    plt.plot(iterations_lst, accs_l2, label="Accuracy of Algorithm without bias")
    plt.xlabel('iterations')
    plt.ylabel('Validation Accuracy (From 0 to 1)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, train_bias_sqres, label="MSE Algorithm with bias")
    plt.plot(iterations_lst, train_l2_sqres, label="MSE Algorithm without bias")
    plt.xlabel('iterations')
    plt.ylabel('Train dataset MSE')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, val_bias_sqres, label="MSE Algorithm with bias")
    plt.plot(iterations_lst, val_l2_sqres, label="MSE Algorithm without bias")
    plt.xlabel('iterations')
    plt.ylabel('Validation dataset MSE')
    plt.legend()
    plt.show()

    acc_l2 = sparse_matrix_evaluate(val_data, reconstruct_matrix_l2)
    print(f"L2 Validation Accuracy is {acc_l2}")
    acc_l2 = sparse_matrix_evaluate(test_data, reconstruct_matrix_l2)
    print(f"L2 Test Accuracy is {acc_l2}")

    acc = sparse_matrix_evaluate(val_data, reconstruct_matrix_bias)
    print(f"Bias Validation Accuracy is {acc}")
    acc = sparse_matrix_evaluate(test_data, reconstruct_matrix_bias)
    print(f"Bias Test Accuracy is {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
