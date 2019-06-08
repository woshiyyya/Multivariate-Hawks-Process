import numpy as np
from math import exp, log
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
from argparse import ArgumentParser

U = [0.001, 0, 0.1, 0.005, 0.007, 0.0025, 0.003, 0.0069, 0.0081, 0.0043]
A = [[0.1, 0.072, 0.0044, 0, 0.0023, 0, 0.09, 0, 0.07, 0.025],
     [0, 0.05, 0.068, 0, 0.027, 0.065, 0, 0, 0.097, 0],
     [0.093, 0, 0.0062, 0.045, 0, 0, 0.053, 0.0095, 0, 0.083],
     [0.019, 0.0033, 0, 0.073, 0.058, 0, 0.056, 0, 0, 0],
     [0.045, 0.091, 0, 0, 0.066, 0, 0, 0.033, 0.0058, 0],
     [0.067, 0, 0, 0, 0, 0.055, 0.063, 0.078, 0.085, 0.0095],
     [0, 0.022, 0.0013, 0, 0.057, 0.091, 0.0088, 0.065, 0, 0.073],
     [0, 0.09, 0, 0.088, 0, 0.078, 0, 0.09, 0.068, 0],
     [0, 0, 0.093, 0, 0.033, 0, 0.069, 0, 0.082, 0.033],
     [0.001, 0, 0.089, 0, 0.008, 0, 0.0069, 0, 0, 0.072]
     ]
w = 0.6
Z = 10
T = 1e5


def intensity(t, dim, history):
    infection_weight = A[dim]
    decay = [[exp(-w * (t - ti)) for ti in seq] for seq in history]
    accumulate_decay = sum([infection_weight[i] * sum(decay[i]) for i in range(Z)])
    return U[dim] + accumulate_decay


def multi_intensity(t, history):
    intensity_arr = np.array([intensity(t, d, history) for d in range(Z)])
    return np.sum(intensity_arr), intensity_arr


def get_candidate(proposal, prob):
    assert proposal < sum(prob)
    tem_prob = 0
    for dim, p in enumerate(prob):
        tem_prob += p
        if proposal < tem_prob:
            return dim


def sample(max_num=100):
    t = 0
    cnt = 0
    history = [[] for _ in range(Z)]
    pbar = tqdm(total=max_num)
    while cnt < max_num:
        prev_lambda, _ = multi_intensity(t, history)
        t += -log(np.random.rand()) / prev_lambda
        proposal = np.random.rand()
        cur_lambda, cur_intensity = multi_intensity(t, history)
        if proposal < cur_lambda / prev_lambda:
            cur_dim = get_candidate(proposal * prev_lambda, cur_intensity)
            history[cur_dim].append(t)
            cnt += 1
            pbar.update(1)
    pbar.close()
    return history


def index_events(history):
    event_seq = sorted(sum(history, []))
    event_mapping = dict()
    for i in range(len(history)):
        for j, t in enumerate(history[i]):
            event_mapping[t] = (i, j)
    return event_seq, event_mapping


def display(history, dim):
    fig, axs = plt.subplots(2, 1)
    x1 = [0] + history[dim]
    y1 = [0] + list(range(0, len(history[dim])))

    x2 = []
    y2 = []

    event_seq, event_mapping = index_events(history)
    total_events = len(event_seq)

    recap_history = [[] for _ in range(Z)]
    for k in range(total_events - 1):
        t, t_next = event_seq[k], event_seq[k + 1]
        i, j = event_mapping[t]
        recap_history[i].insert(j, t)

        sampled_x = np.linspace(t, t_next, 10).tolist()
        sampled_y = []
        for xi in sampled_x:
            sampled_y.append(intensity(xi, dim, recap_history))

        x2.extend(sampled_x)
        y2.extend(sampled_y)

    start_time = min(x2)
    end_time = max(x2)
    x1.append(end_time)
    y1.append(y1[-1])
    x2 = [0, start_time] + x2
    y2 = [0, 0] + y2

    axs[0].step(x1, y1)
    axs[1].plot(x2, y2)
    axs[0].set_xlim(0, end_time)
    axs[1].set_xlim(0, end_time)
    axs[0].set_xlabel('time')
    axs[0].set_ylabel(f'Occurrence N(t)')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel(f'Intensity (dim{dim})')
    axs[0].grid(True)
    axs[1].grid(True)

    fig.tight_layout()
    plt.savefig(f'hawks_dim{dim}.png')
    plt.show()


def gen_data(max_num, max_seqs):
    os.makedirs(f'data/seq{max_num}', exist_ok=True)
    for i in range(max_seqs):
        history = sample(max_num)
        pickle.dump(history, open(f'data/seq{max_num}/sample{max_num}_seq{i}.pkl', 'wb'))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--max_num', type=int, default=20)
    parser.add_argument('--max_seq', type=int, default=20)
    parser.add_argument('--gen_data', action='store_true')
    args = parser.parse_args()

    if args.gen_data:
        gen_data(args.max_num, args.max_seq)
        exit()

    history = sample(args.max_num)
    pickle.dump(history, open(f'data/sample{args.max_num}.pkl', 'wb'))
    for i in range(Z):
        display(history, dim=i)


