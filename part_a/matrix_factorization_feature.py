from utils import *
from scipy.linalg import sqrtm
from matrix_factorization_regularizer_biased import accs_bias, val_bias_sqres, train_bias_sqres, als_bias

import numpy as np
import matplotlib.pyplot as plt

counter_l2 = 0
accs_feature = []
train_feature_sqres = []
val_feature_sqres = []

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

def update_u_z_bias(train_data, lr, u, z, b_u, b_z, lbd, lbd_b, lbd_age, lbd_category, y_age,
                    y_category, student_meta, subject_meta, weights):
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

    b_u_weight = weights[0]
    b_z_weight = weights[1]
    y_age_weight = weights[2]
    y_category_weight = weights[3]

    age = student_meta.get(n, -1)
    age_category = age_classification(age)
    combined_user = u[n, :] + y_age_weight * y_age[age_category, :]
    subject_category = subject_meta.get(q, 9)
    combined_question = z[q, :] + y_category_weight * y_category[subject_category, :]
    prediction = (np.dot(combined_user, combined_question) + b_u_weight * b_u[n] + b_z_weight * b_z[q])

    error = c - prediction

    # Update U and Z based on the error
    u[n, :] += lr * (error * combined_question - lbd * u[n, :])
    z[q, :] += lr * (error * combined_user - lbd * z[q, :])
    b_u[n] += b_u_weight * lr * (error - lbd_b * b_u[n])
    b_z[q] += b_z_weight * lr * (error - lbd_b * b_z[q])
    y_age[age_category, :] += y_age_weight * lr * (error * combined_question) - lbd_age * y_age[age_category, :]
    y_category[subject_category, :] += y_category_weight * lr * (error * combined_user - lbd_category * y_category[subject_category, :])


    ####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z, b_u, b_z, y_age, y_category

def age_classification(age):
    if age == -1:
        return 0
    elif age < 12:
        return 1
    elif age < 14:
        return 2
    elif age < 16:
        return 3
    elif age < 18:
        return 4
    elif age < 20:
        return 5
    elif age < 22:
        return 6
    else:
        return 7

def predict_matrix(u, z, b_u_col, b_z, y_age, y_category, student_meta, subject_meta, weights):
    n, k = u.shape
    m = z.shape[0]
    C = np.zeros((n, m))

    b_u_weight = weights[0]
    b_z_weight = weights[1]
    y_age_weight = weights[2]
    y_category_weight = weights[3]

    for student_id in range(n):
        age = student_meta.get(student_id, -1)
        age_category = age_classification(age)
        combined_user_features = u[student_id, :] + y_age_weight * y_age[age_category, :]
        for item_id in range(m):
            subject_category = subject_meta.get(item_id, -1)
            combined_subject_features = z[item_id, :] + y_category_weight * y_category[subject_category, :]
            C[student_id, item_id] = (np.dot(combined_user_features, combined_subject_features) +
                                                      b_u_weight * b_u_col[student_id][0] + b_z_weight * b_z[item_id])

    return C

def als_feature(train_data, k, lr, num_iteration, lbd, lbd_b, lbd_age, lbd_category, student_meta,
                subject_meta, weights, val_data=None, len_user=None, len_question=None):
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
    y_age = np.random.uniform(low=0.2, high=0.4,
                              size=(8, k))
    y_category = np.random.uniform(low=0.2, high=0.4,
                                   size=(10, k))
    b_u = np.zeros(len_user)
    b_z = np.zeros(len_question)

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(num_iteration):
        u, z, b_u, b_z, y_age, y_category = update_u_z_bias(train_data, lr, u, z, b_u, b_z,
                                                lbd, lbd_b, lbd_age, lbd_category, y_age,
                                                y_category, student_meta, subject_meta, weights)
        if i % 5000 == 0 and val_data:
            b_u_col = b_u.reshape(-1, 1)
            mat = predict_matrix(u, z, b_u_col, b_z, y_age, y_category, student_meta, subject_meta, weights)
            acc = sparse_matrix_evaluate(val_data, mat)
            accs_feature.append(acc)

    b_u_col = b_u.reshape(-1, 1)
    mat = predict_matrix(u, z, b_u_col, b_z, y_age, y_category, student_meta, subject_meta, weights)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    student_meta = load_student_meta_csv("../data")
    subject_meta = load_subject_meta_csv("../data")

    weights = [1, 1, 1, 1]
    reconstruct_matrix_full = als_feature(train_data, 50, 0.025,  150000, 0.025, 0.025,
                                             0.025, 0.025, student_meta, subject_meta, weights=weights)
    reconstruct_matrix_full += 0.0
    # accs_full = accs_feature.copy()
    # globals()['accs_feature'] = []
    # weights = [0.5, 0.5, 1, 1]
    # reconstruct_matrix_null = als_feature(train_data, 50, 0.025, 150000, 0.025, 0.025,
    #                                          0.025, 0.025, student_meta, subject_meta, val_data=val_data,
    #                                          weights=weights)
    # accs_null = accs_feature.copy()
    # globals()['accs_feature'] = []
    # weights = [1, 1, 0.5, 1]
    # reconstruct_matrix_bias = als_feature(train_data, 50, 0.025, 150000, 0.025, 0.025,
    #                                       0.025, 0.025, student_meta, subject_meta, val_data=val_data,
    #                                       weights=weights)
    # accs_bias_only = accs_feature.copy()
    # globals()['accs_feature'] = []
    # weights = [1, 1, 0.5, 0.5]
    # reconstruct_matrix_age = als_feature(train_data, 50, 0.025, 150000, 0.025, 0.025,
    #                                       0.025, 0.025, student_meta, subject_meta, val_data=val_data,
    #                                       weights=weights)
    # accs_age_only = accs_feature.copy()
    # globals()['accs_feature'] = []
    # weights = [1, 1, 1, 0.5]
    # reconstruct_matrix_category = als_feature(train_data, 50, 0.025, 150000, 0.025, 0.025,
    #                                       0.025, 0.025, student_meta, subject_meta, val_data=val_data,
    #                                       weights=weights)
    # accs_category_only = accs_feature.copy()
    #
    # #PART e
    # iterations_lst = list(range(0, 150000, 5000))
    #
    # plt.figure()
    # plt.plot(iterations_lst, accs_full, label="Bias: 1, Age: 1, Question Category: 1")
    # plt.plot(iterations_lst, accs_bias_only, label="Bias: 0.5, Age: 1, Question Category: 1")
    # plt.plot(iterations_lst, accs_age_only, label=" Bias: 1, Age: 0.5, Question Category: 1")
    # plt.plot(iterations_lst, accs_category_only, label="Bias: 1, Age: 0.5, Question Category: 0.5")
    # plt.plot(iterations_lst, accs_null, label="Bias: 1, Age: 1, Question Category: 0.5")
    # plt.xlabel('iterations')
    # plt.ylabel('Validation Accuracy (From 0 to 1)')
    # plt.legend()
    # plt.show()
    #
    # acc_l2 = sparse_matrix_evaluate(val_data, reconstruct_matrix_null)
    # print(f"Base Validation Accuracy is {acc_l2}")
    # acc_l2 = sparse_matrix_evaluate(test_data, reconstruct_matrix_null)
    # print(f"Base Test Accuracy is {acc_l2}")
    #
    # acc = sparse_matrix_evaluate(val_data, reconstruct_matrix_bias)
    # print(f"Bias Validation Accuracy is {acc}")
    # acc = sparse_matrix_evaluate(test_data, reconstruct_matrix_bias)
    # print(f"Bias Test Accuracy is {acc}")
    #
    # acc = sparse_matrix_evaluate(val_data, reconstruct_matrix_age)
    # print(f"Age Validation Accuracy is {acc}")
    # acc = sparse_matrix_evaluate(test_data, reconstruct_matrix_age)
    # print(f"Age Test Accuracy is {acc}")
    #
    # acc = sparse_matrix_evaluate(val_data, reconstruct_matrix_category)
    # print(f"Category Validation Accuracy is {acc}")
    # acc = sparse_matrix_evaluate(test_data, reconstruct_matrix_category)
    # print(f"Category Test Accuracy is {acc}")

    acc = sparse_matrix_evaluate(val_data, reconstruct_matrix_full)
    print(f"Full Validation Accuracy is {acc}")
    acc = sparse_matrix_evaluate(test_data, reconstruct_matrix_full)
    print(f"Full Test Accuracy is {acc}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
