import numpy as np
import gym
from gym import spaces

class MultiAgentPendulumEnv(gym.Env):
    def __init__(self, num_agents=2):
        self.num_agents = num_agents
        self.pendulum_envs = [gym.make('Pendulum-v0') for _ in range(num_agents)]
        self.action_space = [p_env.action_space for p_env in self.pendulum_envs]
        self.observation_space = [p_env.observation_space for p_env in self.pendulum_envs]
        self.seed()

    def reset(self):
        self.states = [env.reset() for env in self.pendulum_envs]
        return np.array(self.states)

    def step(self, actions):
        rewards = []
        dones = []
        infos = []
        for i, env in enumerate(self.pendulum_envs):
            action = actions[i]
            state, reward, done, info = env.step(action)
            if done:
                state = env.reset()
            self.states[i] = state
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.array(self.states), np.array(rewards), np.array(dones), infos

    def render(self, mode='human'):
        for env in self.pendulum_envs:
            env.render(mode=mode)

    def close(self):
        for env in self.pendulum_envs:
            env.close()