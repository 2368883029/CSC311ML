from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt

counter = 0
accs = []
train_sqres = []
val_sqres = []

def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)

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

def update_u_z(train_data, lr, u, z):
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
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]

    ####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, val_data=None, len_user=None, len_question=None):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
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
        u, z = update_u_z(train_data, lr, u, z)
        if i % 1000 == 0 and val_data:
            mat = u @ z.T
            acc = sparse_matrix_evaluate(val_data, mat)
            train_sqre = squared_error_loss(train_data, u, z)
            val_sqre = squared_error_loss(val_data, u, z)
            accs.append(acc)
            train_sqres.append(train_sqre)
            val_sqres.append(val_sqre)

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
    k_set = [1, 5, 10, 20, 40, 80, 160]
    for k in k_set:
        reconstruct_matrix = svd_reconstruct(train_matrix, k)
        acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
        print(f"Accuracy is {acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # Part abc
    # k_set = [4, 8, 16, 32, 50, 64, 80, 100]
    # lr_set = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # iterations = [320000, 64000, 128000, 150000, 200000]
    # params = {}
    # for k in k_set:
    #     for lr in lr_set:
    #         for iteration in iterations:
    #             reconstruct_matrix = als(train_data, k, lr, iteration)
    #             acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
    #             print(f"Final Accuracy is {acc}")
    #             param = f"k:{k} lr:{lr}, iterations:{iteration} acc:{acc}"
    #             params[param] = acc
    #
    # max_key = max(params, key=lambda k: params[k])
    # print(max_key)

    # Part d
    reconstruct_matrix = als(train_data, 50, 0.05, 128000, val_data)
    acc = sparse_matrix_evaluate(val_data, reconstruct_matrix)
    print(f"Validation Accuracy is {acc}")

    # PART e
    iterations_lst = list(range(0, 128000, 1000))

    plt.figure()
    plt.plot(iterations_lst, accs)
    plt.xlabel('iterations')
    plt.ylabel('Validation Accuracy (From 0 to 1)')
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, train_sqres)
    plt.xlabel('iterations')
    plt.ylabel('Train dataset MSE')
    plt.show()

    plt.figure()
    plt.plot(iterations_lst, val_sqres)
    plt.xlabel('iterations')
    plt.ylabel('Validation dataset MSE')
    plt.show()

    acc = sparse_matrix_evaluate(test_data, reconstruct_matrix)
    print(f"Test Accuracy is {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
