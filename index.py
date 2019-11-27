import gym
import numpy as np

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000

epsilon = 0.5
START_EPSILON_DECAYING = 0
END_EPSILON_DECAYING = EPISODES // 3
EPSILON_DECAY_STEP = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

currentEpisode = 0

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)

DISCRETE_OS_WIN_SIZE = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=DISCRETE_OS_SIZE + [env.action_space.n])


def get_discrete_state(state):
    disc_state = (state - env.observation_space.low) / DISCRETE_OS_WIN_SIZE
    return tuple(disc_state.astype(np.int))


for episode in range(EPISODES):

    done = False
    state = env.reset()

    while not done:
        if episode % 500 == 0:
            env.render()

        if episode >= START_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_STEP

        if np.random.random() > epsilon:
            action = np.argmax(q_table[get_discrete_state(state)])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)

        if not done:
            current_q = np.max(q_table[get_discrete_state(state)])
            future_q = np.max(q_table[get_discrete_state(new_state)])

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * future_q)

            q_table[get_discrete_state(state) + (action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[get_discrete_state(state) + (action,)] = 0

        state = new_state

    currentEpisode = currentEpisode + 1

env.close()
