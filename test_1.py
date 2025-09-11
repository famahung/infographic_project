import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------
# 抽數據（嬰兒死亡率）
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

countries = {"CN":"China","IN":"India","US":"United States","JP":"Japan"}
df = fetch_data("SP.DYN.IMRT.IN", countries.keys())
df = df[df["year"]>=1960]

years = sorted(df["year"].unique())
colors = {"China":"#d62828","India":"#f77f00","United States":"#fcbf49","Japan":"#90be6d"}

# ----------------------
# 畫布初始化
# ----------------------
fig, ax = plt.subplots(figsize=(8,8))
ax.set_facecolor("black")
ax.axis("off")

N = 1200
positions = np.random.rand(N,2)*2-1
velocities = np.zeros_like(positions)

# 每個粒子分配國家
country_list = np.array(list(countries.values()))
particle_country = np.random.choice(country_list, N)

title = ax.set_title("", color="white")

# ----------------------
# 自製 noise-like 函數
# ----------------------
def fake_noise(x, y, t):
    return np.sin(1.3*x + 0.7*y + 0.2*t) + np.cos(0.7*x - 1.5*y + 0.1*t)

# ----------------------
# 更新動畫
# ----------------------
def update(frame):
    year = years[frame % len(years)]
    title.set_text(f"Infant Mortality Field · {year}")

    max_rate = df["value"].max()
    rates = {}
    for code, cname in countries.items():
        rate = df[(df["country"]==code)&(df["year"]==year)]["value"].values
        rates[cname] = (rate[0]/max_rate) if len(rate)>0 else 0.1

    global positions, velocities

    # ⭐ 殘影技巧：畫半透明點，而唔係清畫布
    for i in range(N):
        cname = particle_country[i]
        influence = rates[cname]

        x, y = positions[i]
        angle = fake_noise(x*2, y*2, frame*0.05) * np.pi
        dx, dy = np.cos(angle), np.sin(angle)
        velocities[i] = [dx, dy]

        positions[i] += velocities[i]*0.01*(0.5 + influence*2)
        positions[i] = (positions[i] + 2) % 2 - 1  # wrap-around

        ax.plot(positions[i,0], positions[i,1],
                marker='.', color=colors[cname], markersize=1.2, alpha=0.7, linestyle="None")

    return []

ani = animation.FuncAnimation(fig, update, frames=len(years), interval=100, blit=False)
plt.show()

