# KUROSHIO AI — Maritime Chokepoint Congestion Predictor (MVP)
# ------------------------------------------------------------
# This MVP simulates congestion probability at major maritime chokepoints
# using random ship positions and a lightweight PyTorch model.

import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import folium
from streamlit_folium import st_folium
import random

# -----------------------------------------
# 🌊 1. STREAMLIT CONFIG
# -----------------------------------------
st.set_page_config(page_title="Kuroshio AI - Maritime Risk Intelligence", layout="wide")
st.title("🌊 Kuroshio AI — Maritime Chokepoint Congestion Predictor")
st.markdown("#### Predict potential choke point congestion using simulated ship density data.")

# -----------------------------------------
# 🧠 2. SIMPLE PYTORCH MODEL (SIMULATION)
# -----------------------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

model = SimpleModel()

# -----------------------------------------
# 🌍 3. DEFINE MARITIME CHOKEPOINTS
# -----------------------------------------
chokepoints = {
    "Suez Canal": {"entry": (30.585, 32.265), "exit": (29.95, 32.55)},
    "Strait of Hormuz": {"entry": (25.575, 56.250), "exit": (26.5, 57.1)},
    "Panama Canal": {"entry": (9.35, -79.95), "exit": (9.0, -79.6)},
    "Bosporus Strait": {"entry": (41.0, 29.0), "exit": (40.95, 29.1)},
}

# -----------------------------------------
# 🚢 4. HELPER FUNCTIONS
# -----------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def simulate_ships(lat, lon, n=50):
    return [{"lat": lat + random.uniform(-1, 1),
             "lon": lon + random.uniform(-1, 1)} for _ in range(n)]

def count_ships(point, ships, radius_km=25):
    return sum(1 for s in ships if haversine(point[0], point[1], s["lat"], s["lon"]) <= radius_km)

# -----------------------------------------
# ⚙️ 5. RUN SIMULATION
# -----------------------------------------
if st.button("🚀 Run Simulation"):
    data = []
    for name, loc in chokepoints.items():
        ships = simulate_ships(loc["entry"][0], loc["entry"][1], n=np.random.randint(20, 60))
        n_entry = count_ships(loc["entry"], ships)
        n_exit = count_ships(loc["exit"], ships)

        # Predict using PyTorch model
        X = torch.tensor([[n_entry, n_exit]], dtype=torch.float32)
        y_pred = model(X).item()

        severity = "🟢 Low"
        if y_pred > 0.7:
            severity = "🟠 Moderate"
        if y_pred > 0.85:
            severity = "🔴 High"

        data.append({
            "Chokepoint": name,
            "Ships at Entry": n_entry,
            "Ships at Exit": n_exit,
            "Predicted Congestion Probability": round(y_pred, 2),
            "Severity": severity
        })

    # Convert to DataFrame safely
    df = pd.DataFrame(data)

    # -----------------------------------------
    # 📊 6. DISPLAY DASHBOARD
    # -----------------------------------------
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📊 Predicted Congestion Probability")
        st.dataframe(df)
        avg = df["Predicted Congestion Probability"].mean()
        st.metric("🌎 Global Average Congestion", f"{avg:.2f}")

    with col2:
        st.subheader("🗺 Global Chokepoint Visualization")
        world_map = folium.Map(location=[20, 0], zoom_start=2)
        for _, row in df.iterrows():
            name = row["Chokepoint"]
            color = "green"
            if row["Severity"] == "🟠 Moderate":
                color = "orange"
            if row["Severity"] == "🔴 High":
                color = "red"

            entry = chokepoints[name]["entry"]
            exit = chokepoints[name]["exit"]

            folium.Marker(entry, tooltip=f"{name} (Entry)", icon=folium.Icon(color=color)).add_to(world_map)
            folium.Marker(exit, tooltip=f"{name} (Exit)", icon=folium.Icon(color=color)).add_to(world_map)
            folium.PolyLine([entry, exit], color=color, tooltip=f"{name}: {row['Severity']}").add_to(world_map)

        st_folium(world_map, width=900, height=500)

else:
    st.info("👆 Click 'Run Simulation' to generate live congestion predictions.")

# -----------------------------------------
# 🧠 7. FOOTER
# -----------------------------------------
st.markdown("""
---
**Note:**  This MVP uses simulated ship data and a placeholder PyTorch model.  
Future versions will be trained on real AIS, weather, and satellite data.  
Developed by **Kuroshio AI** 🌊
""")
