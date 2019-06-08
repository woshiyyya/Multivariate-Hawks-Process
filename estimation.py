import pickle
import numpy as np
from numpy import exp
from tqdm import trange
from utils import sum_error, Logger
import os
from argparse import ArgumentParser

MAX_DIM = 10
gold_U = np.array([0.001, 0, 0.1, 0.005, 0.007, 0.0025, 0.003, 0.0069, 0.0081, 0.0043])
gold_A = np.array([[0.1, 0.072, 0.0044, 0, 0.0023, 0, 0.09, 0, 0.07, 0.025],
                   [0, 0.05, 0.068, 0, 0.027, 0.065, 0, 0, 0.097, 0],
                   [0.093, 0, 0.0062, 0.045, 0, 0, 0.053, 0.0095, 0, 0.083],
                   [0.019, 0.0033, 0, 0.073, 0.058, 0, 0.056, 0, 0, 0],
                   [0.045, 0.091, 0, 0, 0.066, 0, 0, 0.033, 0.0058, 0],
                   [0.067, 0, 0, 0, 0, 0.055, 0.063, 0.078, 0.085, 0.0095],
                   [0, 0.022, 0.0013, 0, 0.057, 0.091, 0.0088, 0.065, 0, 0.073],
                   [0, 0.09, 0, 0.088, 0, 0.078, 0, 0.09, 0.068, 0],
                   [0, 0, 0.093, 0, 0.033, 0, 0.069, 0, 0.082, 0.033],
                   [0.001, 0, 0.089, 0, 0.008, 0, 0.0069, 0, 0, 0.072]
                   ])
gold_w = np.array(0.6)


def estimate(event_seq, event_mapping, cur_A, cur_U, cur_w):
    N = len(event_seq)
    prob_matrix = np.zeros((N, N))
    for i in range(N):
        di = event_mapping[event_seq[i]][0]
        numerator = np.zeros(N)
        for j in range(i):
            dj = event_mapping[event_seq[j]][0]
            numerator[j] = cur_A[di, dj] * exp(-cur_w * (event_seq[i] - event_seq[j]))
        numerator[i] = cur_U[di]
        prob = numerator / np.sum(numerator)
        prob_matrix[i, :] = prob
    return prob_matrix


def MLE(prob_matrix, event_seq, event_mapping, history, cur_w):
    T = event_seq[-1]
    N = len(event_seq)
    # update U
    new_A = np.zeros((10, 10))
    new_U = np.zeros(10)

    for d in range(MAX_DIM):
        for i, t in enumerate(event_seq):
            if event_mapping[t][0] == d:
                new_U[d] += prob_matrix[i][i]
    new_U /= T

    # update Alpha
    for dim_u in range(MAX_DIM):
        for dim_v in range(MAX_DIM):
            for t_u in history[dim_u]:
                for t_v in history[dim_v]:
                    id_u = event_mapping[t_u][1]
                    id_v = event_mapping[t_v][1]
                    if id_v >= id_u:
                        break
                    new_A[dim_u, dim_v] += prob_matrix[id_u][id_v]
            G = 0
            for t_j in history[dim_v]:
                G += -(exp(-cur_w * (T - t_j)) - 1) / cur_w
            new_A[dim_u, dim_v] /= G

    # update w
    a = b = 0
    for i in range(N):
        for j in range(i):
            a += prob_matrix[i][j]
            b += prob_matrix[i][j] * (event_seq[i] - event_seq[j])
    new_w = a / b
    return new_A, new_U, new_w


def index_events(history):
    # event_mapping = {t: [dim, event_id]}
    event_seq = sorted(sum(history, []))
    event_mapping = dict()
    for i in range(len(history)):
        for t in history[i]:
            event_mapping[t] = [i]
    for tid, t in enumerate(event_seq):
        event_mapping[t].append(tid)
    return event_seq, event_mapping


def EM(data, A, U, w, max_iter=50, log=True):
    best_A = best_U = best_w = None
    best_error = 1e10
    event_seq, event_mapping = index_events(data)
    for _ in trange(max_iter):
        prob_matrix = estimate(event_seq, event_mapping, A, U, w)
        A, U, w = MLE(prob_matrix, event_seq, event_mapping, data, w)
        cur_error = sum_error(A, U, w, gold_A, gold_U, gold_w)

        if log:
            logger.log(A, U, w, gold_A, gold_U, gold_w)

        if cur_error < best_error:
            best_error = cur_error
            best_A = A
            best_U = U
            best_w = w
    return best_A, best_U, best_w, best_error


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_num', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs('data', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    data = pickle.load(open(f'data/sample{args.max_num}.pkl', 'rb'))
    logger = Logger(fn=f'log/{args.max_num}.pkl', history=data)

    np.random.seed(888)
    A = np.random.uniform(0, 1, size=(10, 10))
    U = np.random.uniform(0, 0.1, size=10)
    w = np.random.uniform(0, 1, size=1)

    A, U, w, best_error = EM(data, A, U, w, max_iter=30)
    print("best error = ", best_error)

    logger.store()
