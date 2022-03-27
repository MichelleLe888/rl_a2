import random
import gym
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from collections import deque
import sys

def DqnModel(input_shape, action_space,learning_rate,number_of_nodes = [24,16]):
    input_layer = Input(input_shape)
    layer = Dense(number_of_nodes[0], input_shape=input_shape, activation="relu")(input_layer)
    layer = Dense(number_of_nodes[1], activation="relu")(layer)
    #layer = Dense(64, activation="relu")(layer)
    layer = Dense(action_space, activation="linear")(layer)
    model = Model(inputs=input_layer, outputs=layer)
    model.compile(loss="mse", optimizer=Adam(learning_rate), metrics=["accuracy"])
    return model


class DQNAgent:
    def __init__(self,n_states, n_actions, memory_buffer_size,learning_rate,epsilon,gamma,batch_size, with_memory = True, with_tn = True, number_of_nodes = [24,16]):
        self.state_size = n_states
        self.action_size = n_actions

        self.target_updates = 0
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_updates_treshold = 10

        self.model = DqnModel(input_shape=(self.state_size,), action_space=self.action_size,learning_rate=self.learning_rate,number_of_nodes=number_of_nodes)

        if with_memory:
            self.memory = deque(maxlen=memory_buffer_size)

        if with_tn:
            self.target_network = DqnModel(input_shape=(self.state_size,), action_space=self.action_size, learning_rate=self.learning_rate,number_of_nodes=number_of_nodes)
            self.target_network.set_weights(self.model.get_weights())


    def update_buffer(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict((np.reshape(state, [1,4]))))


    def train(self,terminalstate,transition, with_tn, with_mb):
        if with_mb:

            minibatch = random.sample(self.memory, self.batch_size)
            obses_t = [x[0] for x in minibatch]
            actions = [x[1] for x in minibatch]
            rewards = [x[2] for x in minibatch]
            obses_tp1 = [x[3] for x in minibatch]
            dones = [x[4] for x in minibatch]


            current_qs = self.model.predict(np.array(obses_t),batch_size = self.batch_size)
        else:
            obses_t, action, reward, obses_tp1, done = transition
            current_qs = self.model.predict((np.reshape(obses_t, [1,4])))

        if with_tn:
            if with_mb:
                future_qs = self.target_network.predict((np.array(obses_tp1))) #more stability
            else:
                future_qs = self.target_network.predict((np.reshape(obses_tp1, [1, 4])))

        else:
            if with_mb:
                future_qs = self.model.predict((np.array(obses_tp1)))
            else:
                future_qs = self.model.predict((np.reshape(obses_tp1, [1, 4])))

        if with_mb:
            for i in range(self.batch_size):
                if dones[i]:
                    current_qs[i][actions[i]] = rewards[i]
                else:
                    current_qs[i][actions[i]] = rewards[i] + self.gamma * (np.amax(future_qs[i]))

            self.model.fit(np.array(obses_t), current_qs, batch_size=self.batch_size, verbose=0)

        else:
            if done:
                current_qs[0][action] = reward
            else:
                current_qs[0][action] = reward + self.gamma * (np.amax(future_qs))

            self.model.fit((np.reshape(obses_tp1, [1, 4])), (np.reshape(current_qs, [1, 2])), batch_size=self.batch_size, verbose=0)

        if not terminalstate:
            self.target_updates += 1

        if with_tn:
            if self.target_updates > self.target_updates_treshold:
                self.target_network.set_weights(self.model.get_weights())
                self.target_updates = 0



def deep_q_learning(n_episodes,max_episode_lenght,memory_buffer_size,learning_rate,gamma,epsilon,batch_size,with_mb = True, with_tn = True, number_of_nodes = None):
    env = gym.make('CartPole-v1')
    env.reset()
    DQL_Agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,memory_buffer_size, learning_rate,epsilon, gamma,batch_size,with_memory=with_mb,with_tn=with_tn, number_of_nodes=number_of_nodes)
    rewards = []
    episodes_left = n_episodes
    for episode in range(n_episodes):
        print(episodes_left)
        state = env.reset()
        done = False
        for i in range(max_episode_lenght):
            #env.render()
            action = DQL_Agent.select_action(state)
            next_state, reward, done, _ = env.step(action)



            if with_mb:
                DQL_Agent.update_buffer(state, action, reward, next_state, done)

            transition = (state, action, reward, next_state, done)

            if with_mb:
                if len(DQL_Agent.memory) > batch_size:
                    DQL_Agent.train(done,transition,with_tn=with_tn, with_mb=with_mb)
            else:
                DQL_Agent.train(done, transition, with_tn=with_tn, with_mb=with_mb)


            state = next_state

            if done or i == (max_episode_lenght-1):
                rewards.append(i+1)
                episodes_left-=1
                break

    env.close()
    return rewards

def main(argv):
    with_tn = False
    with_mb = False

    if len(sys.argv) > 0:
        if "target_network" in sys.argv:
            with_tn = True
        if "experience_buffer" in sys.argv:
            with_mb = True

    print("Deep Q learning: target_network: {}, experience buffer {}".format(with_tn, with_mb))
    n_episodes = 200
    memory_buffer_size = 2000
    gamma = 0.95
    learning_rate = 0.01
    max_episode_lenght = 200
    epsilon = 0.2
    batch_size = 32

    rewards = deep_q_learning(n_episodes=n_episodes,max_episode_lenght=max_episode_lenght,memory_buffer_size=memory_buffer_size,
                              learning_rate=learning_rate, gamma=gamma, epsilon=epsilon,batch_size=batch_size, with_mb=with_mb,with_tn=with_tn )

    x = range(0, len(rewards))
    fig, ax = plt.subplots()
    ax.plot(x,rewards)

    z = np.polyfit(x, rewards, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")

    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.show()


    title = "Deep Q learning performance"
    if with_tn or with_mb:
        title += " with "
        if with_tn:
            title+= "Target Network "
        if with_tn and with_mb:
            title += "and "
        if with_mb:
            title+= "Experience Buffer "

    plt.title(title)

    print("Obtained rewards: {}".format(rewards))


if __name__ == "__main__":
    main(sys.argv)

