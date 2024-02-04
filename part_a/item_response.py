from utils import *

import numpy as np
import matplotlib.pyplot as plt

num_user = 0
num_questions = 0

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))

def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    user_id = data.get('user_id')
    question_id = data.get('question_id')
    is_correct = data.get('is_correct')
    sum = 0

    for k in range(len(user_id)):
        c_ij = is_correct[k]
        i = user_id[k]
        j = question_id[k]
        theta_i = theta[i]
        beta_j = beta[j]
        p_correct_log = np.log(np.exp(theta_i - beta_j) / (np.exp(theta_i - beta_j) + 1))
        sum += (1 - c_ij) * (1 - p_correct_log) + c_ij * p_correct_log
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -sum

def partial_theta(data, theta, beta):
    """
    Compute the partial derivative of log-likelihood respect to theta
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: Vector
    """
    user_id = data.get('user_id')
    question_id = data.get('question_id')
    is_correct = data.get('is_correct')
    derivative_theta = np.zeros(num_user)

    for k in range(len(user_id)):
        c_ij = is_correct[k]
        i = user_id[k]
        j = question_id[k]
        theta_i = theta[i]
        beta_j = beta[j]

        nume = (c_ij - 1) * np.exp(theta_i) + c_ij * np.exp(beta_j)
        deno = np.exp(theta_i) + np.exp(beta_j)

        derivative_theta[i] += nume / deno

    return derivative_theta

def partial_beta(data, theta, beta):
    """
    Compute the partial derivative of log-likelihood respect to beta
    """
    user_id = data.get('user_id')
    question_id = data.get('question_id')
    is_correct = data.get('is_correct')
    derivative_beta = np.zeros(num_questions)

    for k in range(len(user_id)):
        c_ij = is_correct[k]
        i = user_id[k]
        j = question_id[k]
        theta_i = theta[i]
        beta_j = beta[j]

        nume = -((c_ij - 1) * np.exp(theta_i) + c_ij * np.exp(beta_j))
        deno = np.exp(theta_i) + np.exp(beta_j)

        derivative_beta[j] += nume / deno

    return derivative_beta

def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    new_theta = theta + lr * partial_theta(data, theta, beta)
    new_beta = beta + lr * partial_beta(data, theta, beta)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return new_theta, new_beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(num_user)
    beta = np.zeros(num_questions)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    globals()['num_user'], globals()['num_questions'] = sparse_matrix.shape
    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc_lst_train = irt(train_data, val_data, 0.005, 30)
    print(f"The validation accuracy is {val_acc_lst_train[-1]}")
    score = evaluate(data=test_data, theta=theta, beta=beta)
    print(f"The test accuracy is {score}")
    iterations = np.arange(1, 31)

    plt.figure()
    plt.plot(iterations, val_acc_lst_train)
    plt.xlabel('iterations')
    plt.ylabel('Accuracy (From 0 to 1)')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    beta_j1 = beta[0]
    beta_j2 = beta[1]
    beta_j3 = beta[2]

    theta_values = np.linspace(-2, 2, 100)

    prob_j1 = np.exp(theta_values - beta_j1) / (1 + np.exp(theta_values - beta_j1))
    prob_j2 = np.exp(theta_values - beta_j2) / (1 + np.exp(theta_values - beta_j2))
    prob_j3 = np.exp(theta_values - beta_j3) / (1 + np.exp(theta_values - beta_j3))

    plt.figure()
    plt.plot(theta_values, prob_j1, label=f'Question j1 with beta: {beta_j1}')
    plt.plot(theta_values, prob_j2, label=f'Question j2 with beta: {beta_j2}')
    plt.plot(theta_values, prob_j3, label=f'Question j3 with beta: {beta_j3}')

    plt.xlabel('Student ability theta')
    plt.ylabel('Probability of Correct Response p(c_ij | theta, beta)')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
