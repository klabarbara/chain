from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
import glob
import csv
import math

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
with open('data/entities_updated.csv', 'r', encoding='utf-8') as f:
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
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def distance_between_dea(dea_a, dea_b):
    lat1, lon1 = dea_info[dea_a]['lat'], dea_info[dea_a]['lon']
    lat2, lon2 = dea_info[dea_b]['lat'], dea_info[dea_b]['lon']
    return haversine_distance(lat1, lon1, lat2, lon2)

# Step 4: Build transitions from processed jsonl files
transitions = {}
for file_path in glob.glob('data/processed_with_ndc/*.jsonl'):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            path = entry['path']
            for i in range(len(path)-1):
                a, b = path[i], path[i+1]
                dist = distance_between_dea(a, b)
                if a not in transitions:
                    transitions[a] = []
                transitions[a].append((b, dist))

nodes = list(dea_info.keys())
node_to_idx = {n: i for i, n in enumerate(nodes)}


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

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_t = torch.tensor([state], device=device)
        with torch.no_grad():
            q_values = dqn(state_t)
        return torch.argmax(q_values).item()

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for t in range(max_steps_per_episode):
        action = choose_action(state, epsilon)
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
