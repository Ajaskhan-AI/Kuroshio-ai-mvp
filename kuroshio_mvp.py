# ============================================================
# 🌊 KUROSHIO AI – CHOKEPOINT CONGESTION PREDICTOR (MVP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn

# -----------------------------------------
# 🔧 1. PAGE CONFIG
# -----------------------------------------
st.set_page_config(page_title="Kuroshio AI – Chokepoint Congestion Predictor", layout="wide")
st.title("🌊 Kuroshio AI – Chokepoint Congestion Predictor")
st.markdown("""
This MVP uses a **PyTorch model** to simulate congestion probabilities  
at global maritime chokepoints based on entry/exit ship densities.
""")

# -----------------------------------------
# 🌍 2. DEFINE CHOKEPOINTS
# -----------------------------------------
chokepoints = {
    "Suez Canal": {"entry": (30.585, 32.265), "exit": (29.966, 32.567)},
    "Strait of Hormuz": {"entry": (26.575, 56.250), "exit": (25.300, 57.050)},
    "Strait of Malacca": {"entry": (2.550, 101.000), "exit": (5.978, 95.936)},
    "Panama Canal": {"entry": (9.350, -79.920), "exit": (9.200, -79.700)},
    "Bosporus Strait": {"entry": (41.000, 29.000), "exit": (40.950, 29.100)}
}

# -----------------------------------------
# ⚙️ 3. DEFINE A SIMPLE PYTORCH MODEL
# -----------------------------------------
class CongestionPredictor(nn.Module):
    def __init__(self):
        super(CongestionPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize model
model = CongestionPredictor()

# -----------------------------------------
# ⚙️ 4. SIMULATE SHIP DATA
# -----------------------------------------
def simulate_ships(lat, lon, n=30, spread=0.5):
    ships = []
    for _ in range(n):
        ships.append({
            "lat": lat + np.random.uniform(-spread, spread),
            "lon": lon + np.random.uniform(-spread, spread),
            "speed": np.random.uniform(5, 25)
        })
    return ships

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def count_ships(point, ships, radius_km=25):
    return sum(1 for s in ships if haversine(point[0], point[1], s["lat"], s["lon"]) <= radius_km)

# -----------------------------------------
# ⚙️ 5. RUN SIMULATION
# -----------------------------------------
data = []
for name, loc in chokepoints.items():
    ships = simulate_ships(loc["entry"][0], loc["entry"][1], n=np.random.randint(20, 60))
    n_entry = count_ships(loc["entry"], ships)
    n_exit = count_ships(loc["exit"], ships)

    # Prepare input tensor for PyTorch model
    X = torch.tensor([[n_entry, n_exit]], dtype=torch.float32)
    y_pred = model(X).item()

    severity = "🟢 Low"
    if y_pred > 0.7: severity = "🟠 Moderate"
    if y_pred > 0.85: severity = "🔴 High"

    data.append({
        "Chokepoint": name,
        "Ships at Entry": n_entry,
        "Ships at Exit": n_exit,
        "Predicted Congestion Probability": round(y_pred, 2),
        "Severity": severity
    })

df = pd.DataFrame(data)

# -----------------------------------------
# 📊 6. DISPLAY DASHBOARD
# -----------------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Predicted Congestion Probability")
    st.dataframe(df.style.background_gradient(cmap="YlOrRd", subset=["Predicted Congestion Probability"]))
    avg = df["Predicted Congestion Probability"].mean()
    st.metric("🌎 Global Average Congestion", f"{avg:.2f}")

with col2:
    st.subheader("🗺️ Global Chokepoint Visualization")

    world_map = folium.Map(location=[20, 0], zoom_start=2)
    for _, row in df.iterrows():
        name = row["Chokepoint"]
        color = "green"
        if row["Severity"] == "🟠 Moderate": color = "orange"
        if row["Severity"] == "🔴 High": color = "red"
        entry = chokepoints[name]["entry"]
        exit = chokepoints[name]["exit"]

        folium.Marker(entry, tooltip=f"{name} (Entry)", icon=folium.Icon(color=color)).add_to(world_map)
        folium.Marker(exit, tooltip=f"{name} (Exit)", icon=folium.Icon(color=color)).add_to(world_map)
        folium.PolyLine([entry, exit], color=color, tooltip=f"{name}: {row['Severity']}").add_to(world_map)

    st_folium(world_map, width=900, height=500)

# -----------------------------------------
# 🧠 7. FOOTER
# -----------------------------------------
st.markdown("""
---
**Note:**  
This MVP uses simulated ship data and a placeholder PyTorch model.  
Future versions will be trained on real AIS, weather, and satellite data.  
Developed by **Kuroshio AI** 🌊
""")
