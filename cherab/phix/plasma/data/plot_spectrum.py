from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# import spectrum data
data = np.loadtxt(
    Path(__file__).parent.resolve() / "spectrum_phix_discharge.TXT",
    delimiter=";",
    skiprows=8,
)
wavelength, samples = data[:, 0], data[:, 1]
peaks, _ = find_peaks(samples, prominence=1000, distance=7)

# plot
plt.rcParams["font.size"] = 14
plt.figure()
plt.plot(wavelength, samples)
# plt.plot(wavelength[peaks], samples[peaks]),'x')
plt.xlabel("wavelength [nm]")
plt.xlim([390, 790])
plt.xticks(wavelength[peaks], rotation=90)
plt.grid(axis="x")
plt.tight_layout()

# annotation
# peak line label
label_line = [
    "$H_{\delta}$",
    "$H_{\gamma}$",
    r"$H_{\beta}$",
    "Ar II, C I, or Fe I",
    "Fe II, O IV, Ar I",
    "C I , Fe, O II ",
    "Ar, C I, O III ",
    r"$H_{\alpha}$",
    r"$C III (3d\rightarrow3p)$",
]
for txt, x, y in zip(label_line, wavelength[peaks], samples[peaks]):
    plt.annotate(txt, (x, y), horizontalalignment="right", rotation=90)

plt.show()
