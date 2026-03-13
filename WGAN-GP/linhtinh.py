import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("WGAN_GP_training_history.csv")

df = df[200:]  # bỏ warm-up

df["critic_smooth"] = df["critic_loss"].rolling(100).mean()
df["generator_smooth"] = df["generator_loss"].rolling(100).mean()

epochs = np.arange(200, 200 + len(df))

plt.figure(figsize=(8,5))

plt.plot(epochs, df["critic_smooth"], label="Critic loss")
plt.plot(epochs, df["generator_smooth"], label="Generator loss")

plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Smoothed training loss curves")

# trục X chi tiết hơn
plt.xticks(np.arange(0, 6001, 500))

# trục Y chi tiết
plt.yticks(np.arange(-1.4, 0, 0.1))
plt.ylim(-1.4, 0)

plt.legend()
plt.tight_layout()

plt.savefig("smoothed_training_loss.svg", format="svg", bbox_inches="tight")