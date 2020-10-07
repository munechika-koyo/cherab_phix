# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy.io import loadmat
import cv2
from raysect.optical import World
from cherab.phix.plasma import TSCEquilibrium
from cherab.phix.tools import import_phix_rtm, show_phix_profile
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = 12
plt.rcParams["figure.dpi"] = 200


# ------------------- PATH ------------------------
CHERAB_PHIX_DIR = os.path.dirname(__file__).split("demos")[0]

CAMERA_DATA_PATH = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data")
FRAME_DIR = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data/frames")
RECONST_DIR = os.path.join(CHERAB_PHIX_DIR, "output/data/experiment/shot_17393/camera_data/reconstraction")
GIF_SAVE_DIR = os.path.join(RECONST_DIR, "only_reconst")

# -------------- Raysect & Cherab init -----------------
world = World()
eq = TSCEquilibrium(folder="phix10")
rtm = import_phix_rtm(world, eq)
# ------------------------------------------------------

# ----------------- Limiter position shot_17393 ----------------
limiter_original = np.array([[131, 140], [133, 433], [44, 422], [6, 334], [6, 240], [44, 153], [131, 140]])
center = np.array([128, 256])
limiter = np.zeros_like(limiter_original)
for i in range(len(limiter_original[:, 0])):
    limiter[i, :] = 2 * np.array([center[0], 0]) + np.array([[-1, 0], [0, 1]]) @ limiter_original[i, :]
# ---------------------------------------------------------------

# frames list
list_frame = os.listdir(RECONST_DIR)
list_frame = [frame_name for frame_name in list_frame if ".npy" in frame_name]

# extract max value
vmax_reconst = 0.0
for frame_name in list_frame:
    reconst = np.load(os.path.join(RECONST_DIR, frame_name))
    vmax_buff = reconst.max()
    if vmax_buff > vmax_reconst:
        vmax_reconst = vmax_buff

# frames iteration
for frame_name in list_frame:
    print(f"frame: {frame_name}")
    fig, ax = plt.subplots()

    frame_name = "1091.npy"
    reconst_profile = np.load(os.path.join(RECONST_DIR, frame_name))
    show_phix_profile(
        ax, reconst_profile, rtm=rtm, vmax=vmax_reconst, vmin=0.0, levels=np.linspace(2e+13, vmax_reconst, 15)
    )
    ax.set_title(f'frame: {frame_name.split(".")[0]}')
    ax.set_xlabel("$R$[m]")
    ax.set_ylabel("$Z$[m]")
    plt.show()
    # save
    fig.savefig(os.path.join(GIF_SAVE_DIR, f'{frame_name.split(".")[0]}.png'), bbox_inches="tight")

    plt.close(fig)

# generate gif
os.system(
    f'convert -delay 30 -loop 0 {os.path.join(GIF_SAVE_DIR, "*.png")} {os.path.join(GIF_SAVE_DIR, "shot_17393.gif")}'
)

# camera image & reconst image
# vmax_reconst = 0.0
# vmax_camera = 0.0
# for frame_name in list_frame:
#     reconst = np.load(os.path.join(RECONST_DIR, frame_name))
#     camera_image = loadmat(os.path.join(FRAME_DIR, frame_name.split(".")[0] + ".mat"))["unshiftedIm"][:, :, 0]
#     vmax_reconst_buff = reconst.max()
#     vmax_camera_buff = camera_image.max()
#     if vmax_reconst_buff > vmax_reconst:
#         vmax_reconst = vmax_reconst_buff
#     if vmax_camera_buff > vmax_camera:
#         vmax_camera = vmax_camera_buff

# for frame_name in list_frame:
#     print(f"frame: {frame_name}")
#     frame = eval(frame_name.split(".")[0])
#     camera_image_R = loadmat(os.path.join(FRAME_DIR, f"{frame}.mat"))["unshiftedIm"][:, :, 0]
#     reconst_image = np.load(os.path.join(RECONST_DIR, f"{frame}.npy"))

#     # fig, axes = plt.subplots(1, 2)
#     fig, axes = plt.subplots(nrows=2, ncols=1, tight_layout=True)
#     axes[0].imshow(np.fliplr(camera_image_R), vmax=vmax_camera, cmap="inferno", extent=(0, 255, 511, 0))
#     axes[0].plot(limiter[:, 0] - 1, limiter[:, 1] - 1, color="w", linewidth=0.5)
#     axes[0].xaxis.set_visible(False)
#     axes[0].yaxis.set_visible(False)
#     axes[0].yaxis.set_ticklabels([])
#     # axes[0].set_title("camera image")
#     # axes[0].set_xlabel("x [px]")
#     # axes[0].set_ylabel("y [px]")
#     show_phix_profile(
#         axes[1], reconst_image, rtm=rtm, vmin=0, vmax=vmax_reconst, levels=np.linspace(0, vmax_reconst, 15)
#     )
#     # axes[1].set_title("Reconstruction image")
#     axes[1].set_xlabel("$R$[m]")
#     axes[1].set_ylabel("$Z$[m]")
#     # fig.suptitle(f"frame: {frame}")
#     axes[0].set_title(f"frame: {frame}")
#     fig.set_size_inches(3, 6)
#     # fig.subplots_adjust(hspace=0, wspace=0.35)
#     fig.subplots_adjust(hspace=-0.25, wspace=0.1)
#     # save
#     # plt.show()
#     fig.savefig(os.path.join(GIF_SAVE_DIR, f'{frame_name.split(".")[0]}.png'))

#     plt.close(fig)

# # generate gif
# os.system(
#     f'convert -delay 20 -loop 0 {os.path.join(GIF_SAVE_DIR, "*.png")} {os.path.join(GIF_SAVE_DIR, "shot_17393.gif")}'
# )
