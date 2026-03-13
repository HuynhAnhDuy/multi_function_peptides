import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("WGAN_GP_training_history.csv")

epochs = range(1, len(df) + 1)

# ===== Training losses =====
plt.figure(figsize=(8,5))

plt.plot(epochs, df["critic_loss"], label="Critic loss")
plt.plot(epochs, df["generator_loss"], label="Generator loss")

epochs = np.arange(200, 200 + len(df))

plt.xlabel("Epoch", fontsize=12, fontweight='bold',fontstyle='italic')
plt.ylabel("Loss value", fontsize=12, fontweight='bold',fontstyle='italic')


# trục X chi tiết hơn
plt.xticks(np.arange(0, 6001, 500))

# trục Y chi tiết
plt.yticks(np.arange(-3, 0.6, 0.2))
plt.ylim(-3, 0.6)
plt.legend()

plt.tight_layout()

plt.savefig("training_losses.svg", format="svg")


# ===== Wasserstein distance =====
plt.figure(figsize=(8,5))

plt.plot(epochs, df["wasserstein"], label="Wasserstein estimate")

plt.xlabel("Epoch", fontsize=12, fontweight='bold',fontstyle='italic')
plt.ylabel("Estimated Wasserstein distance", fontsize=12, fontweight='bold',fontstyle='italic')

plt.legend()
plt.tight_layout()

plt.savefig("wasserstein_curve.svg", format="svg")