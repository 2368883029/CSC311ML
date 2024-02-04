from utils import *
from scipy.linalg import sqrtm
from matrix_factorization import als, train_sqres, val_sqres, accs

import numpy as np
import matplotlib.pyplot as plt

counter_l2 = 0
accs_l2 = []
train_l2_sqres = []
val_l2_sqres = []

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

def update_u_z_L2constraint(train_data, lr, u, z, lbd):
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

    error = c - np.dot(u[n], z[q])

    # Update U and Z based on the error
    u[n] += lr * (error * z[q] - lbd * u[n])
    z[q] += lr * (error * u[n] - lbd * z[q])

    ####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als_L2constraint(train_data, k, lr, num_iteration, lbd, val_data=None, len_user=None, len_question=None):
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

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z = update_u_z_L2constraint(train_data, lr, u, z, lbd)
        if i % 10000 == 0 and val_data:
            mat = u @ z.T
            acc = sparse_matrix_evaluate(val_data, mat)
            train_sqre = squared_error_loss(train_data, u, z)
            val_sqre = squared_error_loss(val_data, u, z)
            accs_l2.append(acc)
            train_l2_sqres.append(train_sqre)
            val_l2_sqres.append(val_sqre)

    mat = u @ z.T
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
    reconstruct_matrix_l2 = als_L2constraint(train_data, 50, 0.05, 400000, 0.025, val_data=val_data)
    reconstruct_matrix = als(train_data, 50, 0.05, 400000, val_data=val_data)

    # PART e
    iterations_lst = list(range(0, 400000, 10000))

    plt.figure()
    plt.plot(iterations_lst, accs_l2, label="Accuracy of Algorithm with L2 regularization")
    plt.plot(iterations_lst, accs, label="Accuracy of Algorithm without L2 regularization")
    plt.xlabel('iterations')
    plt.ylabel('Validation Accuracy (From 0 to 1)')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, train_l2_sqres, label="MSE Algorithm with L2 regularization")
    plt.plot(iterations_lst, train_sqres, label="MSE Algorithm without L2 regularization")
    plt.xlabel('iterations')
    plt.ylabel('Train dataset MSE')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, val_l2_sqres, label="MSE Algorithm with L2 regularization")
    plt.plot(iterations_lst, val_sqres, label="MSE Algorithm without L2 regularization")
    plt.xlabel('iterations')
    plt.ylabel('Validation dataset MSE')
    plt.legend()
    plt.show()

    acc_l2 = sparse_matrix_evaluate(val_data, reconstruct_matrix_l2)
    print(f"L2 Validation Accuracy is {acc_l2}")
    acc_l2 = sparse_matrix_evaluate(test_data, reconstruct_matrix_l2)
    print(f"L2 Test Accuracy is {acc_l2}")

    acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
    print(f"Validation Accuracy is {acc}")
    acc = sparse_matrix_evaluate(test_data, reconstruct_matrix)
    print(f"Test Accuracy is {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
