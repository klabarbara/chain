from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
import math
import pandas as pd

from model import DQN
from rl_environment import DistributionPathEnv

# Step 1: Load ZIP-to-lat/lon CSV
zip_coords = {}
with open('data/zip_ll.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        z = row['ZIP'].strip()
        lat = float(row['LAT'])
        lon = float(row['LNG'])
        zip_coords[z] = (lat, lon)

# Step 2: Load DEA info and map DEA codes to lat/lon
dea_info = {}
with open('data/cleaned_entities.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dea_code = row['dea_no']
        z = row['zip']
        # Get lat/lon from zip_coords if available, else skip or handle missing
        if z in zip_coords:
            lat, lon = zip_coords[z]
        else:
            lat, lon = None, None
        dea_info[dea_code] = {
            'bus_act': row['bus_act'],
            'city': row['city'],
            'state': row['state'],
            'zip': z,
            'lat': lat,
            'lon': lon
        }

# Step 3: Define a haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return float('inf')  # If missing coords, distance is infinite (or handle differently)
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def distance_between_dea(dea_a, dea_b):
    lat1, lon1 = dea_info[dea_a]['lat'], dea_info[dea_a]['lon']
    lat2, lon2 = dea_info[dea_b]['lat'], dea_info[dea_b]['lon']
    return haversine_distance(lat1, lon1, lat2, lon2)

# Step 4: Build transitions from cleaned_distribution_paths.csv
transitions = {}
paths_df = pd.read_csv('data/cleaned_distribution_paths.csv')

# Flatten the paths into transitions
for _, row in paths_df.iterrows():
    from_dea = row['from_dea_no']
    to_dea = row['to_dea_no']
    drug = row['drug']
    quantity = row['quantity']
    # compute distance from lat/lon if needed
    dist = haversine_distance(row['from_lat'], row['from_lon'], row['to_lat'], row['to_lon'])
    
    # Skip paths where DEA nodes have no valid coordinates
    if from_dea not in dea_info or to_dea not in dea_info:
        continue
    if dea_info[from_dea]['lat'] is None or dea_info[to_dea]['lat'] is None:
        continue
    
    # Avoid duplicate or zero-distance transitions
    if from_dea == to_dea:
        continue  # Skip identical consecutive nodes

    dist = distance_between_dea(from_dea, to_dea)
    if dist == 0.0:
        continue  # Skip transitions with zero distance

    # Add to transitions only if it's unique
    if from_dea not in transitions:
        transitions[from_dea] = []
    if (to_dea, dist) not in transitions[from_dea]:
        transitions[from_dea].append((to_dea, dist, quantity, drug))

print(f"Number of nodes in transitions: {len(transitions)}")
for k, v in list(transitions.items())[:5]:  # Print sample transitions
    print(f"From {k}: {v}")

nodes = list(dea_info.keys())
node_to_idx = {n: i for i, n in enumerate(nodes)}

# Step 5: Set up the RL environment and DQN
env = DistributionPathEnv(nodes, transitions, node_to_idx)
num_episodes = 1000
max_steps_per_episode = 20
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
memory = deque(maxlen=10000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dqn = DQN(num_nodes=len(nodes)).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=lr)

# Step 6: RL Training
def choose_action(state, epsilon, dqn, env, nodes, transitions, node_to_idx):
    # Identify valid actions from the current node
    current_node = nodes[state]  # Convert state (index) to DEA code
    if current_node in transitions:
        valid_next_nodes = [node_to_idx[t[0]] for t in transitions[current_node]]
    else:
        # If no valid transitions, just return random action or handle differently
        # But we ensured this shouldn’t happen at reset, still let's handle gracefully:
        return env.action_space.sample()

    if random.random() < epsilon:
        # Choose a random valid action
        action = random.choice(valid_next_nodes)
    else:
        # Choose the best valid action according to Q-values
        state_t = torch.tensor([state], device=device)
        with torch.no_grad():
            q_values = dqn(state_t)  # shape (1, num_nodes)
        
        # Mask invalid actions by setting them to a very large negative number
        q_values_masked = q_values.clone()
        
        # Create a mask for all actions
        invalid_actions = set(range(len(nodes))) - set(valid_next_nodes)
        for ia in invalid_actions:
            q_values_masked[0, ia] = -1e9  # Large negative value

        action = torch.argmax(q_values_masked).item()
    return action

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for t in range(max_steps_per_episode):
        action = choose_action(state, epsilon, dqn, env, nodes, transitions, node_to_idx)
        next_state, reward, done, info = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        if done:
            break

    # Training step
    if len(memory) >= batch_size:
        batch = random.sample(memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(states, device=device)
        actions_t = torch.tensor(actions, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states_t = torch.tensor(next_states, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

        # Compute Q targets
        q_values = dqn(states_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = dqn(next_states_t).max(1)[0]
            q_target = rewards_t + gamma * q_next * (1 - dones_t)

        loss = F.mse_loss(q_values, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")
