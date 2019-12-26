
import tensorflow as tf
import gym
import numpy as np
from keras import *
from keras import Sequential
from keras.layers import Dense
import random
import time
import statistics

# game environment
env = gym.make('Acrobot-v1')
#number_of_actions = env.action_space # three discrete valid actions
number_of_actions = 3
print("Number of actions", number_of_actions)
observation_shape = env.observation_space.shape[0] # valid current_states are an array of 6 numbers
print("Observation sensors", observation_shape)


# hyper-parameters
number_of_episodes = 10
EPSILON = 0.2
DECAYED_EPSILON = 0.1
DECAY = 0.1
ALPHA = 0.1 # learning rate
GAMMA = 0.9 # discount factor
TAU = 0.01 # target network soft update hyperparameter
BATCH_SIZE = 100
MEMORY_LIMIT = 500000
MIN_MEMORY_FOR_UPDATE = 50
MEMORY = []


# Deep neural net function
def neural_net():
	nn = Sequential()
	nn.add(Dense(64, activation='relu', input_dim = observation_shape))
	nn.add(Dense(20, activation = 'relu'))
	nn.add(Dense(number_of_actions, activation='linear'))
	return nn

 # Eperience replay used for learning
def replay(primary_model, target_model):
	minibatch = random.sample(MEMORY, BATCH_SIZE)
	minibatch_updated_Q_values = []
	# for each experience in minibatch calculate new Q values for each state and action pair
	for observation in minibatch:
		current_state, action, reward, next_state, done = observation
		current_state = np.reshape(current_state, (1, observation_shape))
		observation_updated_Q_values = primary_model.predict(current_state)[0]
		if done:
			Q_update = reward
		else:
			next_state = np.reshape(next_state, (1, observation_shape))
			# select action
			action_selected = np.argmax(primary_model.predict(next_state))
			# evaluate action
			value_estimated = target_model.predict(next_state)[0][action_selected]
			# Bellman equation
			Q_update = reward + GAMMA * value_estimated
		observation_updated_Q_values[action] = Q_update
		minibatch_updated_Q_values.append(observation_updated_Q_values)
	minibatch_states = np.array([e[0] for e in minibatch])
	minibatch_updated_Q_values = np.array(minibatch_updated_Q_values)
	primary_model.fit(minibatch_states, minibatch_updated_Q_values, verbose=False, epochs=1)


def update_target_network(primary_model, target_model):
	# model parameters of primary network that is being trained
	primary_model_theta = primary_model.get_weights()
	# model parameters of target network
	target_model_theta = target_model.get_weights()
	# update target model parameters based on primary model parameters
	counter = 0
	for q_weight, target_weight in zip(primary_model_theta,target_model_theta):
		# soft target network update
		target_weight = target_weight * (1-TAU) + q_weight * TAU
		target_model_theta[counter] = target_weight
		counter += 1
    # set new model paramenters
	target_model.set_weights(target_model_theta)
	return target_model


# EPSILON greedy policy
def choose_action(state, EPSILON):
	# choose action
	if np.random.uniform() < EPSILON:
		# exploratory random action
		action = env.action_space.sample() 
	else:
		# select action that yields highest estimated value
		state = np.reshape(state, (1, observation_shape)) 
		q_values = primary_model.predict(state)[0]
		action = np.argmax(q_values)
	return action

# Double Q learning algorithm
def double_Q_learning(number_of_episodes, EPSILON):
	total_reward = []
	total_time = []
	for episode in range(number_of_episodes): 
		print("-------------- Starting new episode -------------------")
		# initialize state
		current_state = env.reset() 
		# initialize score per episode
		episode_reward = 0
		#for timestep in range(number_of_timesteps):
		timestep = 1
		# time required per episode
		start_time = time.time()
		while True:
			# render environment
			env.render()
			# decay EPSILON over time
			if EPSILON > DECAYED_EPSILON:
				EPSILON = EPSILON * DECAY
			# choose action
			action = choose_action(current_state, EPSILON)
			print("----------------- Action taken at timestep", timestep, ":", action, "----------------------")
			# take action
			next_state, reward, done, info = env.step(action) # reward = float, done = Bool
			episode_reward += reward
			# remember
			if MEMORY_LIMIT > len(MEMORY):
				MEMORY.append((current_state, action, reward, next_state, done))
			# update state
			current_state = next_state
			if len(MEMORY) > BATCH_SIZE:
				replay(primary_model, target_model)
			if len(MEMORY) > MIN_MEMORY_FOR_UPDATE:
				update_target_network(primary_model, target_model)

			if done:
				print("The goal was reached and episode finished after", timestep, "timesteps")
				break
			timestep += 1
		total_reward.append(episode_reward)
		time_per_episode = time.time() - start_time
		total_time.append(time_per_episode)
	env.close()
	return total_reward, total_time


# Primary neural net that will be trained
primary_model = neural_net()
primary_model.compile(loss='mse', optimizer='adam')

# Target neural net parameters are not trained via back-propagation – instead they are periodically copied from the primary network
target_model = neural_net()

# Algorithm 
result = double_Q_learning(number_of_episodes, EPSILON)

# Results
print("Results: cummulative reward per episode: ")
print(result[0])
print("Average cummulative reward: ", statistics.mean(result[0]))
print("Results: training time per episode: ")
print(result[1])
print("Average training time: ", statistics.mean(result[1]))


