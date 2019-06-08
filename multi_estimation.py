from estimation import *
from utils import Logger
import numpy as np
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--max_num", type=int, default=1000)
    args = parser.parse_args()

    np.random.seed(888)

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

    A_avg = 0
    U_avg = 0
    w_avg = 0
    A_list, U_list, w_list, error_list = [], [], [], []

    for seq in trange(20):
        data = pickle.load(open(f'data/seq{args.max_num}/sample{args.max_num}_seq{seq}.pkl', 'rb'))
        A = np.random.uniform(0, 1, size=(10, 10))
        U = np.random.uniform(0, 1, size=10)
        w = np.random.uniform(0, 1, size=1)
        A, U, w, best_error = EM(data, A, U, w, max_iter=20, log=False)
        A_list.append(A)
        U_list.append(U)
        w_list.append(w)
        error_list.append(best_error)

    best_n = 5
    selected_ids = np.argsort(error_list)[:best_n]
    for i in selected_ids:
        print(error_list[i])
        A_avg += A_list[i] / best_n
        U_avg += U_list[i] / best_n
        w_avg += w_list[i] / best_n
    best_error = sum_error(A_avg, U_avg, w_avg, gold_A, gold_U, gold_w)
    print("best error: ", best_error)
    pickle.dump([A_avg, U_avg, w_avg, best_error], open(f'data/ckpt/{args.max_num}_Ensemble_20.pkl', 'wb'))
    pickle.dump([A_list, U_list, w_list, error_list], open(f'data/ckpt/{args.max_num}_All_20.pkl', 'wb'))
