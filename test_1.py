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
    "China": "#de2910",         # Red (China flag)
    "India": "#ff9933",         # Saffron (India flag)
    "United States": "#3c3b6e",  # Blue (US flag)
    "Japan": "#ffffff"          # White (Japan flag)
}

# ----------------------
# Figure & Axis
# ----------------------
fig = plt.figure(figsize=(960/100, 540/100))  # 960x540 pixels
fig.patch.set_facecolor("black")
ax = fig.add_subplot(111, projection="3d", facecolor="black")

# remove everything (ticks, axes, cube box)
ax.set_axis_off()

# ----------------------
# Particles
# ----------------------
N = 50  # further reduced for vortex effect
# Initialize particles in a ring for vortex
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
radius = 1.5 + 1.0 * np.random.rand(N)
z = (np.random.rand(N) - 0.5) * 1.2
positions = np.stack([
    radius * np.cos(theta),
    radius * np.sin(theta),
    z
], axis=1)
velocities = np.zeros_like(positions)

country_list = np.array(list(countries.values()))
particle_country = np.random.choice(country_list, N)

# trail buffer
trail_length = 30  # longer for smoother lines
history = [[] for _ in range(N)]

# ----------------------
# Noise field
# ----------------------


def fake_noise(x, y, z, t):
    # Simple noise for subtle vertical swirl
    return np.sin(2 * x + 2 * y + 0.5 * t) + np.cos(1.5 * z + 0.3 * t)

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
    # system spin
    R = rotation_matrix(np.radians(frame * 0.5))
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

    # Camera movement: slowly orbit around the vortex
    azim = 30 + (frame * 360 / 250)  # 360-degree orbit over 10s
    elev = 30 + 10 * np.sin(frame * 2 * np.pi / 250)  # gentle up/down
    ax.view_init(elev=elev, azim=azim)

    for i in range(N):
        cname = particle_country[i]
        influence = rates[cname]

        x, y, z = positions[i]
        # Vortex/circular motion around center
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        # Circular velocity
        v_theta = 0.13 + 0.09 * influence
        # Vertical swirl
        v_z = 0.04 * fake_noise(x, y, z, frame * 0.04 + i * 0.02)
        # Outward/inward breathing
        v_r = 0.01 * np.sin(frame * 0.02 + i * 0.1)
        # Update position in polar coordinates
        new_theta = theta + v_theta
        new_r = r + v_r
        new_z = z + v_z
        positions[i] = [
            new_r * np.cos(new_theta), new_r * np.sin(new_theta), new_z]

        # apply rotation
        positions[i] = positions[i] @ R.T

        # manage trails
        history[i].append(positions[i].copy())
        if len(history[i]) > trail_length:
            history[i].pop(0)

        # Depth of field: fade and size by z
        xs, ys, zs = zip(*history[i])
        z_depth = (zs[-1] + 1) / 2  # normalize z to [0,1]
        dot_alpha = 0.4 + 0.6 * (1 - z_depth)
        dot_size = 6 + 18 * (1 - z_depth)

        # Smooth trail: interpolate points for curve
        from scipy.interpolate import splprep, splev
        if len(xs) > 3:
            tck, u = splprep([xs, ys, zs], s=0)
            u_fine = np.linspace(0, 1, 30)
            x_smooth, y_smooth, z_smooth = splev(u_fine, tck)
            for j in range(len(x_smooth) - 1):
                trail_z = (z_smooth[j] + 1) / 2
                alpha = (j + 1) / len(x_smooth)
                width = 1.2 + 1.5 * (1 - trail_z) * (alpha ** 1.5)
                ax.plot(x_smooth[j:j+2], y_smooth[j:j+2], z_smooth[j:j+2],
                        color=colors[cname], alpha=alpha * 0.7 * (1 - trail_z * 0.5), linewidth=width)
        else:
            for j in range(len(xs) - 1):
                trail_z = (zs[j] + 1) / 2
                alpha = (j + 1) / len(xs)
                width = 1.2 + 1.5 * (1 - trail_z) * (alpha ** 1.5)
                ax.plot(xs[j:j+2], ys[j:j+2], zs[j:j+2],
                        color=colors[cname], alpha=alpha * 0.7 * (1 - trail_z * 0.5), linewidth=width)

        # draw dot with depth of field
        ax.scatter(xs[-1], ys[-1], zs[-1],
                   c=colors[cname], s=dot_size, alpha=dot_alpha)

    return []


# ----------------------
# Run
# ----------------------
ani = animation.FuncAnimation(
    fig, update, frames=250, interval=40, blit=False)  # 250 frames, ~10s at 25fps
plt.show()

# Save as GIF (make sure pillow is installed)
ani.save("output.gif", writer="pillow", fps=25, writer_kwargs={"loop": 0})
