import matplotlib.pyplot as plt
import pickle
import numpy as np
from os.path import join


def plot_esb():
    # [[0.452, 0.388],
    #  [0.441, 0.327],
    #  []]
    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0, 'right': 1, 'left': -1}

        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

    N = 5
    single_best = np.array([0.452, 0.441, 0.322, 0.313, 0.239])
    ensemble = np.array([0.388, 0.327, 0.213, 0.192, 0.119])
    fig, ax = plt.subplots()
    width = 0.35
    rect1 = ax.bar(np.arange(5) - width/2, single_best, width=width)
    rect2 = ax.bar(np.arange(5) + width/2, ensemble, width=width)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(['N=100', 'N=200', 'N=500', 'N=800', 'N=1000'])
    ax.legend(labels=['single-seq best', 'multi-seq ensemble'])

    autolabel(rect1, "center")
    autolabel(rect2, "center")
    ax.set_title("Relative Error in Single/Multi-seq Settings")
    plt.show()
    fig.savefig('img/ensemble.jpg')


def plot_EM(fn='log'):
    # MAXNUM = [100, 500, 800, 1000, 2000, 3000]
    MAXNUM = [100, 500, 800, 1000, 3000, 5000]
    log = {"error": [], "nll": []}
    for n in MAXNUM:
        data = pickle.load(open(join(fn, f"{n}.pkl"), 'rb'))
        print(min(data['error']))
        log['error'].append(data['error'])
        log['nll'].append(data['nll'])
        print(len(data['nll']))
        print(np.argsort(data['error']))

    exit()
    print(len(log['error']))
    plt.figure(0)
    plt.plot(np.array(log['error']).transpose())
    plt.legend(labels=MAXNUM, loc='upper right')
    plt.xticks(np.linspace(0, 30, 7))
    plt.title("EM Parameter Errors")
    plt.xlabel("n_iter")
    plt.ylabel("error")
    plt.savefig('img/EM-Errors.jpg')
    plt.show()

    input()
    plt.figure(1)
    plt.plot(np.array(log['nll']).transpose())
    plt.legend(labels=MAXNUM, loc='upper right')
    plt.xticks(np.linspace(0, 30, 31))
    plt.title("EM Parameter NLL")
    plt.xlabel("n_iter")
    plt.ylabel("negative log likelihood")
    plt.savefig('img/EM-NLL.jpg')
    plt.show()


if __name__ == "__main__":
    # plot_EM('log')
    plot_esb()