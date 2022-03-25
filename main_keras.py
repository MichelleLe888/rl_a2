import gym
import time
import numpy as np
from collections import deque
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt

def example():
    env = gym.make("CartPole-v1")
    obs = env.reset(seed=42)

    for episode in range(50):
        obs = env.reset()
        for t in range(100):
            env.render()
            #print(observation)
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            time.sleep(0.001)

            if done:
                print("Episode {} finished after {} timesteps".format(episode ,t + 1))
                break
        #observation, info = env.reset(return_info=True)
    env.close()

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size[0]
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.learning_rate = 0.001
        self.model = self.model(state_size)
        self.results = []

    def model(self,state_size):
        model = Sequential()
        model.add(Dense(24,input_dim=self.state_size))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.add(Activation('linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    #epsilon greedy policy
    def get_action(self, q_values, epsilon=0.1):
        if np.random.uniform() < epsilon:
            return np.random.randint(0,self.action_size)
        else:
            return np.argmax(q_values)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        # sampling random transitions from the whole batch
        for state, action, reward, next_state, done in minibatch:
            #if terminal yj is rj
            target = reward

            #if not-terminal
            # yj is rj + gamma * maxaction(q(nextstate,nextaction)
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])

            #performing gradient descent
            #calculate current q-value for current state
            target_f = self.model.predict(state)

            #update q-value for current action with new targets
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)


if __name__ == "__main__":

    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    agent = DQN(env.observation_space.shape,2)
    episodes = 100
    batch_size=32
    scores=[]

    # Iterate the game
    for e in range(episodes):

        # reset state in the beginning of each game
        state = env.reset(seed=33)
        #print(state.shape)
        state = np.reshape(state, [1, 4])
        #print(state.shape)


        max_score = 200
        for time_t in range(max_score):
            #env.render()

            act_values = agent.model.predict(state)
            action = agent.get_action(act_values[0])
            print("action",action)

            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            # memorize the previous state, action, reward, and done
            agent.memorize(state, action, reward, next_state, done)

            state = next_state

            if done:
                scores.append(time_t+1)
                print("episode: {}/{}, score: {}"
                      .format(e+1, episodes, time_t+1))
                break

            if len(agent.memory) > batch_size:
                history = agent.replay(batch_size)

    env.close()
    plt.plot(scores)
    plt.title('Training: batch size of 32')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.show()
