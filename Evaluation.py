import numpy as np
import time
from DQL import deep_q_learning
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


class LearningCurvePlot:

    def __init__(self, title=None):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Duration')
        if title is not None:
            self.ax.set_title(title)

    def add_curve(self, y, label=None):
        if label is not None:
            self.ax.plot(y, label=label)
        else:
            self.ax.plot(y)

    def set_ylim(self, lower, upper):
        self.ax.set_ylim([lower, upper])

    def save(self, name):
        self.ax.legend()
        self.fig.savefig(name, dpi=300)



def smooth(y, window, poly=1):
    return savgol_filter(y,window,poly)


def average_over_repetitions(n_repetitions,n_episodes,smoothing_widnow,with_mb=True,with_tn=True,learning_rate=None,epsilon=None,number_of_nodes = [24,16]):
    reward_results = np.empty([n_repetitions, n_episodes])  # Result array
    for rep in range(n_repetitions):
        #     rewards = deep_q_learning(n_episodes=n_episodes,max_episode_lenght=max_episode_lenght,memory_buffer_size=memory_buffer_size, learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,batch_size=64, with_mb=True,with_tn=True )
        rewards = deep_q_learning(n_episodes=n_episodes,max_episode_lenght=200,memory_buffer_size=2000,learning_rate=0.1,gamma=1.0,epsilon=0.01,batch_size=32,with_mb=with_mb,with_tn=with_tn,number_of_nodes = number_of_nodes)
        reward_results[rep] = rewards

    learning_curve = np.mean(reward_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_widnow)  # additional smoothing
    return learning_curve

def experiment():
    n_episodes = 200
    Plot = LearningCurvePlot()
    learning_curve = average_over_repetitions(30, n_episodes,31,with_mb=True,with_tn=True,learning_rate=0.1)
    Plot.add_curve(learning_curve)
    Plot.save("network_learning.png")

def experiment_lr():
    n_episodes = 200
    learning_rates = [0.01, 0.2, 0.5]
    Plot = LearningCurvePlot()
    for lr in learning_rates:
        learning_curve = average_over_repetitions(30, n_episodes,smoothing_widnow=31,with_mb=True,with_tn=True,learning_rate=lr)
        Plot.add_curve(learning_curve,label=r'Learning rate = {}'.format(lr))
    Plot.save("learning_rates.png")
def experiment_eb():
    n_episodes = 150
    smoothing_window = 31
    Plot = LearningCurvePlot()
    now = time.time()
    without = average_over_repetitions(1,n_episodes,smoothing_window,with_mb=False,with_tn=False)
    print('Without Experience Buffer and Target Network: {} minutes'.format((time.time() - now) / 60))
    now = time.time()
    with_eb_learning_curve = average_over_repetitions(1,n_episodes,smoothing_window,with_mb=True,with_tn=False)
    print('With Experience Buffer: {} minutes'.format((time.time() - now) / 60))
    now = time.time()
    with_tn_learning_curve = average_over_repetitions(1,n_episodes,smoothing_window,with_mb=False,with_tn=True)
    print('With Target Network: {} minutes'.format((time.time() - now) / 60))
    now = time.time()
    with_tn__eb_learning_curve = average_over_repetitions(1,n_episodes,smoothing_window, with_mb=True, with_tn=True)
    print('With Target Network and Experience Buffer: {} minutes'.format((time.time() - now) / 60))

    Plot.add_curve(without,label=r'Deep Q learning without Replay Buffer and Target Network')
    Plot.add_curve(with_eb_learning_curve,label=r'Deep Q learning with Replay Buffer')
    Plot.add_curve(with_tn__eb_learning_curve,label=r'Deep Q learning with Replay Buffer and Target Network')
    Plot.add_curve(with_tn_learning_curve,label=r'Deep Q learning with Target Network')

    Plot.save("different_elements.png")

def experiment_archtecture():
    n_episodes = 200
    smoothing_window = 31
    Plot = LearningCurvePlot()
    number_of_nodes = [[24,16],[8,6],[256,64]]
    for non in number_of_nodes:
        print(non)
        learning_curve = average_over_repetitions(1, n_episodes,smoothing_widnow=smoothing_window,with_mb=True,with_tn=True,number_of_nodes = non)
        Plot.add_curve(learning_curve,label=r'Number of nodes = {}'.format(non))
    Plot.save("number_of_nodes.png")


def experiment_expl():
    epsilons = [0.01, 0.2, 1]
    for eps in epsilons:
        n_episodes = 100
        Plot = LearningCurvePlot()
        learning_curve = average_over_repetitions(1, n_episodes, 31, with_mb=True, with_tn=True, epsilon=eps)
        Plot.add_curve(learning_curve,label=r'Epsilon = {}'.format(eps))
    Plot.save("exploration_rates.png")


#experiment()
#experiment_lr()
experiment_eb()
#experiment_expl()
#experiment_archtecture()