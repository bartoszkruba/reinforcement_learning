import gym
import numpy as np

env = gym.make("Pendulum-v0")

done = False

env.reset()

while not done:
    env.render()
    action = 0
    new_state, reward, done, _ = env.step(env.action_space.sample())

env.close()
