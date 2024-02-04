from utils import *
from scipy.linalg import sqrtm
from matrix_factorization import *

import numpy as np
import matplotlib.pyplot as plt


def bagging_sparse_matrix_evaluate(data, matrices, threshold=0.5):
    """ Given the sparse matrix represent, return the accuracy of the prediction on data.

    :param data: A dictionary {user_id: list, question_id: list, is_correct: list}
    :param matrices: Array of 2D matrices
    :param threshold: float
    :return: float
    """
    total_prediction = 0
    total_accurate = 0
    total_matrix = len(matrices)
    for i in range(len(data["is_correct"])):
        positive_vote = 0
        negative_vote = 0
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        for matrix in matrices:
            if matrix[cur_user_id, cur_question_id] >= threshold:
                positive_vote += 1
            if matrix[cur_user_id, cur_question_id] < threshold:
                negative_vote += 1

        if positive_vote >= negative_vote and data["is_correct"][i]:
            total_accurate += 1
        if positive_vote < negative_vote and not data["is_correct"][i]:
            total_accurate += 1
        total_prediction += 1
    return total_accurate / float(total_prediction)

def bootstrap_sample(data):
    indices = np.random.choice(len(data['user_id']), size=len(data['user_id']), replace=True)

    for key in data:
        data[key] = np.array(data[key])[indices]

    return data

def main():
    # 1. Randomly picks data from training set
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    reconstruct_matrices = []
    len_user = len(set(train_data["user_id"]))
    len_question = len(set(train_data["question_id"]))

    # k_set = [16, 32, 50, 64]
    # lr_set = [0.0005, 0.001, 0.005, 0.01, 0.05]
    # iterations = [32000, 64000, 128000, 150000]
    # params = {}
    # for k in k_set:
    #     for lr in lr_set:
    #         for iteration in iterations:
    #             for _ in range(3):
    #                 sample_train_data = bootstrap_sample(train_data)
    #                 reconstruct_matrix = als(sample_train_data, k, lr, iteration, len_user=len_user, len_question=len_question)
    #                 reconstruct_matrices.append(reconstruct_matrix)
    #             acc = bagging_sparse_matrix_evaluate(val_data, reconstruct_matrices)
    #             print(f"k:{k} lr:{lr}, iterations:{iteration} acc:{acc}")
    #             param = f"k:{k} lr:{lr}, iterations:{iteration} acc:{acc}"
    #             params[param] = acc
    #             reconstruct_matrices = []

    # max_key = max(params, key=lambda k: params[k])
    # print(max_key)

    for _ in range(3):
        sample_train_data = bootstrap_sample(train_data)

        reconstruct_matrix = als(sample_train_data, 50, 0.05, 128000, len_user=len_user, len_question=len_question)
        reconstruct_matrices.append(reconstruct_matrix)

    val_acc = bagging_sparse_matrix_evaluate(val_data, reconstruct_matrices)
    test_acc = bagging_sparse_matrix_evaluate(test_data, reconstruct_matrices)
    print(f"Validation Acc is {val_acc}")
    print(f"Test Acc is {test_acc}")

if __name__ == "__main__":
    main()
