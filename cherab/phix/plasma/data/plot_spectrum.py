# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from raysect.optical.colour import ciexyz_to_srgb, ciexyz_x, ciexyz_y, ciexyz_z
from scipy.signal import find_peaks

# %%
# import spectrum data
# --------------------
data = np.loadtxt(
    Path(__file__).parent.resolve() / "spectrum_phix_discharge.TXT",
    delimiter=";",
    skiprows=8,
)
wavelength, samples = data[:, 0], data[:, 1]
peaks, _ = find_peaks(samples, prominence=1000, distance=7)

# %%
# plot
# ----
plt.rcParams["font.size"] = 15

fig, ax = plt.subplots(dpi=150, figsize=(13, 5), constrained_layout=True)
ax.plot(wavelength, samples)
ax.set_xlabel("wavelength [nm]")
ax.set_ylabel("intensity [a.u.]")
ax.set_xlim(390, 790)
ax.set_ylim(-5e2, 2e4)

# annotate peak lines
peak_line_labels = [
    "Hδ",
    "Hγ",
    "Hβ",
    "Ar II, C I, or Fe I",
    "Fe II, O IV, Ar I",
    "C I , Fe, O II ",
    "Ar, C I, O III ",
    "Hα",
    r"C III $(3d\rightarrow3p)$",
]
for label, x in zip(peak_line_labels, wavelength[peaks]):
    # set srgb color from spectrum
    cie_x = ciexyz_x(x) * 106.8566
    cie_y = ciexyz_y(x) * 106.8566
    cie_z = ciexyz_z(x) * 106.8566
    srgb = ciexyz_to_srgb(cie_x, cie_y, cie_z)

    # plot peak lines
    ax.axvline(x=x, linewidth=0.75, linestyle="--", color="k", alpha=0.3, zorder=-1)

    # annotate peak lines
    text = f"{label}  {x:.1f} nm"
    x -= 3 if "Hα" in text else 0
    align = "left" if "493" in text else "right"
    ax.annotate(text, (x, 3000), horizontalalignment=align, rotation=90, color=srgb)

plt.show()
