import numpy as np
from simulation import index_events
import pickle


class Logger:
    def __init__(self, fn, history):
        self.history = history
        self.total_events = sum([len(d) for d in history])
        self.record = dict()
        self.record['error'] = []
        self.record['nll'] = []
        self.record['parameter'] = []
        self.fn = fn
        print("TOTAL EVENTS: ", self.total_events)

    def log(self, A, U, w, gold_A, gold_U, gold_w):
        error = sum_error(A, U, w, gold_A, gold_U, gold_w)
        nll = -log_likelihood(alpha=A, w=w, mu=U, history=self.history) / self.total_events
        self.record['nll'].append(nll)
        self.record['error'].append(error)
        self.record['parameter'].append((A, U, w))
        print(f"NLL     :{nll}, Error     :{error}")

    def store(self):
        pickle.dump(self.record, open(self.fn, 'wb'))


def distance(pred, gold):
    pred = pred.reshape(-1)
    gold = gold.reshape(-1)
    error = 0
    assert pred.shape == gold.shape
    for i, j in zip(gold, pred):
        if i != 0:
            error += abs(i - j) / i
        else:
            error += abs(i - j)
    return error


def sum_error(A, U, w, gold_A, gold_U, gold_w):
    error = 0
    error += distance(A, gold_A)
    error += distance(U, gold_U)
    error += distance(w, gold_w)
    return error / (A.size + U.size + 1)


def log_likelihood(alpha, w, mu, history):
    event_seq, event_mapping = index_events(history)

    T = event_seq[-1]
    term1 = 0
    term2 = 0
    term3 = 0

    for dim_i in range(10):
        term1 += -mu[dim_i] * T
        for dim_j in range(10):
            alpha_i_j = alpha[dim_i][dim_j]
            term2 += -alpha_i_j * (sum([1 - np.exp(-w * (T - t)) for t in history[dim_j]]) / w)

    prev_dim = []
    prev_time = []
    for time in event_seq:
        dim = event_mapping[time][0]
        term3 += \
            np.log(mu[dim] + sum([alpha[dim][dim_n] * np.exp(-w * (time - time_n)) for dim_n, time_n in zip(prev_dim, prev_time)]))
        prev_dim.append(dim)
        prev_time.append(time)
    return term1 + term2 + term3



