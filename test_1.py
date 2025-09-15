import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ----------------------
# Fetch data (infant mortality)
# ----------------------


def fetch_data(indicator, countries):
    dfs = []
    for c in countries:
        api = f"https://api.worldbank.org/v2/country/{c}/indicator/{indicator}?format=json&per_page=5000"
        res = requests.get(api).json()
        records = [
            {"country": c, "year": int(r["date"]), "value": r["value"]}
            for r in res[1] if r["value"] is not None
        ]
        dfs.extend(records)
    return pd.DataFrame(dfs)


countries = {"CN": "China", "IN": "India",
             "US": "United States", "JP": "Japan"}
df = fetch_data("SP.DYN.IMRT.IN", countries.keys())
df = df[df["year"] >= 1960]

years = sorted(df["year"].unique())
colors = {
    "China": "#d62828",
    "India": "#f77f00",
    "United States": "#fcbf49",
    "Japan": "#90be6d"
}

# ----------------------
# Figure & Axis
# ----------------------
fig = plt.figure(figsize=(10, 10))
fig.patch.set_facecolor("black")
ax = fig.add_subplot(111, projection="3d", facecolor="black")

# remove everything (ticks, axes, cube box)
ax.set_axis_off()

# ----------------------
# Particles
# ----------------------
N = 300
positions = np.random.rand(N, 3) * 2 - 1
velocities = np.zeros_like(positions)

country_list = np.array(list(countries.values()))
particle_country = np.random.choice(country_list, N)

# trail buffer
trail_length = 15
history = [[] for _ in range(N)]

# ----------------------
# Noise field
# ----------------------


def fake_noise(x, y, z, t):
    return (
        np.sin(1.2 * x + 0.8 * y + 0.6 * z + 0.2 * t)
        + np.cos(1.1 * x - 0.7 * y + 0.3 * z + 0.15 * t)
    )

# ----------------------
# Rotation helper
# ----------------------


def rotation_matrix(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

# ----------------------
# Update animation
# ----------------------


def update(frame):
    year = years[frame % len(years)]

    max_rate = df["value"].max()
    rates = {}
    for code, cname in countries.items():
        rate = df[(df["country"] == code) & (
            df["year"] == year)]["value"].values
        rates[cname] = (rate[0] / max_rate) if len(rate) > 0 else 0.1

    global positions, velocities, history
    ax.cla()
    ax.set_facecolor("black")
    ax.set_axis_off()  # ensure no axes/box come back

    # rotate the camera
    ax.view_init(elev=25, azim=frame * 0.8)

    # system spin
    R = rotation_matrix(np.radians(frame * 0.5))

    for i in range(N):
        cname = particle_country[i]
        influence = rates[cname]

        x, y, z = positions[i]
        angle = fake_noise(x * 2, y * 2, z * 2, frame * 0.05) * np.pi
        dx, dy, dz = np.cos(angle), np.sin(angle), np.sin(angle * 0.7)
        velocities[i] = [dx, dy, dz]

        positions[i] += velocities[i] * 0.01 * (0.5 + influence * 2)
        positions[i] = (positions[i] + 2) % 2 - 1

        # apply rotation
        positions[i] = positions[i] @ R.T

        # manage trails
        history[i].append(positions[i].copy())
        if len(history[i]) > trail_length:
            history[i].pop(0)

        # draw trails fading
        xs, ys, zs = zip(*history[i])
        for j in range(len(xs) - 1):
            alpha = (j + 1) / len(xs)
            ax.plot(xs[j:j+2], ys[j:j+2], zs[j:j+2],
                    color=colors[cname], alpha=alpha * 0.7, linewidth=0.8)

        # draw dot
        ax.scatter(xs[-1], ys[-1], zs[-1],
                   c=colors[cname], s=8, alpha=0.9)

    return []


# ----------------------
# Run
# ----------------------
ani = animation.FuncAnimation(
    fig, update, frames=len(years), interval=80, blit=False)
plt.show()

ani.save("output.gif", writer="pillow", fps=20)
