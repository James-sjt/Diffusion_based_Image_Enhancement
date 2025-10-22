import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Valid_log_Model.csv")  # training_log_UNet.csv, Valid_log_Model.csv
for line in df.itertuples():
    print(line)

plt.plot(df["epoch"], df["avg_loss"], label="Avg Loss")
plt.plot(df["epoch"], df["avg_ddpm"], label="ddpm Loss")
plt.plot(df["epoch"], df["avg_x0_latent"], label="x0 Latent Loss")
plt.plot(df["epoch"], df["avg_recon_aux"], label="Recon Aux Loss")
plt.plot(df["epoch"], df["avg_perc"], label="Perceptual Loss")
plt.plot(df["epoch"], df["avg_lap"], label="Laplacian Loss")
plt.plot(df["epoch"], df["avg_patch_perc"], label="Patch Perception Loss")
plt.plot(df["epoch"], df["avg_fft"], label="FFT Loss")
plt.plot(df["epoch"], df["avg_contrast"], label="Contrast Loss")
plt.plot(df["epoch"], df["avg_local_contrast"], label="Local Contrast Loss")
plt.plot(df["epoch"], df["lr"], label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
