import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Q-learning

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n) # taula de #estats x #accions per guardar els Q-valors
gamma = 0.999

# You can change these values to see how they affect the results
lr=0.01 # learning rate (alpha)
epsilon = 0.2 # exploration probability (epsilon-greedy policy)
#epsilon = 1

G = 0
print('Training...')

for episode in range(1, 10001):

    # TODO: Implement the Q-learning algorithm that decides actions using the epsilon-greedy policy
    state = env.reset() 
    done = None
    while done != True:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        next_state, reward, done, _, info = env.step(action)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += lr * td_delta
        state = next_state
        G += reward

    
    # End TODO

    # Every 500 episodes, print the average collected reward during training (stored in variable G)
    if episode % 500 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G/500), 'Epsilon =', epsilon, 'lr =', lr)
        G=0

print('End training...')


print('Visualizing policy...')
# Create and print policy and V for valid states
policy = np.zeros((env.world_row,env.world_col))
V = np.zeros((env.world_row,env.world_col))
policy[0,3]=-1  # Special value for terminal state
policy[1,3]=-1  # Special value for terminal state
policy[1,1]=-1  # Not defined for non-valid states

state = np.nditer(env.map, flags=['multi_index'])
while not state.finished:
    if env.map[state.multi_index]==0:
        policy[state.multi_index] = np.argmax(Q[env.cell_id(state.multi_index)])
        V[state.multi_index]= np.max(Q[env.cell_id(state.multi_index)])
    state.iternext()

print_policy(policy,V)


print('Visualizing episodes...')
# Let's test your policy!
for i in range(10):
    state = env.reset()
    env.render()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()


print('Collecting reward of the policy while testing (no exploration)...')
G=0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(1000):
    state = env.reset()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G = G + reward
print('Average reward of the policy while testing:', G/1000)

# hacer 10 vecer para ver si simple da el mismo (optimo)
# ver si las politicas son razonables (no va directo al estado malo)
# epsilon 0.2 vol dir 20% de les vegades explora, provar valors més petits o baixant amb el temps i valorar si els resultats milloren o son més estables
# provar amb diferents lr (0.01 és gran), més petit triga més pero més estable
# diferencia performance entre train i test
# mirar si los resultados de politica son similares a los metodos de programacion dinamica (lab 2)
# diferencia entre on-policy (sarsa) i off-policy (q-learning) (eliminar la desviacion en grid.py)
# més castigo, linea 82 de grid.py
# si montecarlos se parece más a sarsa o a q-learning
# si epsilon a 1 en q-learning aprende puramente con acciones aleatorias, pero la politica final es la optima (siempre) (no pasa en sarsa ni en montecarlo)
# 