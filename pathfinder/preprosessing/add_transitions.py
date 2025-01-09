import pandas as pd
import math

# Step 1: Data loading: 'dea_info' for mapping DEA -> lat/lon

dea_info = {}
entities_df = pd.read_csv("../../data/cleaned_entities.csv") 
for _, row in entities_df.iterrows():
    dea_code = row["dea_no"]
    lat = row["lat"]
    lon = row["lon"]
    dea_info[dea_code] = {
        "lat": lat,
        "lon": lon
    }

# Step 2: Define distance functions

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Returns the haversine distance in kilometers between two coords.
    If any coordinate is None, returns float('inf').
    """
    if None in (lat1, lon1, lat2, lon2):
        return float('inf') 
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         + math.sin(dlon/2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def distance_between_dea(dea_a, dea_b):
    """
    Convenience function to look up lat/lon from dea_info,
    then compute haversine distance.
    """
    lat1, lon1 = dea_info[dea_a]['lat'], dea_info[dea_a]['lon']
    lat2, lon2 = dea_info[dea_b]['lat'], dea_info[dea_b]['lon']
    return haversine_distance(lat1, lon1, lat2, lon2)

# Step 3: Build transitions from cleaned_distribution_paths.csv

paths_df = pd.read_csv("../../data/cleaned_distribution_paths.csv")

transitions = {}
for _, row in paths_df.iterrows():
    from_dea = row["from_dea_no"]
    to_dea = row["to_dea_no"]
    drug = row["drug"]
    quantity = row["quantity"]

    dist = distance_between_dea(from_dea, to_dea)

    # Skip missing coords
    if from_dea not in dea_info or to_dea not in dea_info:
        continue
    if dea_info[from_dea]["lat"] is None or dea_info[to_dea]["lat"] is None:
        continue

    # Skip zero-distance or identical nodes
    if from_dea == to_dea or dist == 0.0:
        continue

    if from_dea not in transitions:
        transitions[from_dea] = []
    # Store a tuple with the extra info
    # Check if we already have that exact (to_dea, dist, quantity, drug)
    if (to_dea, dist, quantity, drug) not in transitions[from_dea]:
        transitions[from_dea].append((to_dea, dist, quantity, drug))

print(f"Number of from-nodes in transitions: {len(transitions)}")

# Step 4: Export the transitions to a CSV

# Generating a long-form table with columns [from_dea, to_dea, distance, quantity, drug]
# TODO: do I still need this?
rows = []
for from_dea, edges in transitions.items():
    for (to_dea, dist, quantity, drug) in edges:
        rows.append({
            "from_dea_no": from_dea,
            "to_dea_no": to_dea,
            "distance": dist,
            "quantity": quantity,
            "drug": drug
        })

transitions_df = pd.DataFrame(rows)
transitions_df.to_csv("transitions.csv", index=False)

print(f"Saved {len(transitions_df)} transitions to 'transitions.csv'.")
