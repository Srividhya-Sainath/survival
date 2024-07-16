from collections import namedtuple
import os
import glob
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle
import h5py
import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from mil.ViT import ViT
import openslide
import numpy.typing as npt
from scipy import interpolate
from pathlib import Path

def vals_to_im(scores, coords):
    """
    Args:
        coords: need to be normalized such that adjacent tiles' coordinates differ by one
    """
    size = coords.max(0)[::-1] + 1
    if scores.ndimension() == 1:
        im = np.zeros(size)
    elif scores.ndimension() == 2:
        im = np.zeros((*size, scores.size(-1)))
    else:
        raise ValueError(f"{scores.ndimension()=}")
    for score, c in zip(scores, coords):
        x, y = c[0], c[1]
        im[y, x] = score.cpu().detach().numpy()
    return im

learner_path = "/mnt/bulk-io/vidhya/karishma/DACHS/TFMS-UNI-BATCH64-404020/lr_1e-05_l1_0.001_l2_0.001_best_model.pth"
feature_name_pattern = "/mnt/bulk-io/vidhya/karishma/CIRCULATE/UNI-FEATURES/E2E_macenko_uni_old/*.h5"
image_name_pattern = "/mnt/bulk-io/vidhya/karishma/CIRCULATE/IMGS/*.ndpi"
scores_csv = "/mnt/bulk-io/vidhya/karishma/DACHS/TFMS-UNI-BATCH64-404020/CIRCULATE_score.csv"
output_folder = "/home/vidhya/marugoto/marugoto/survival/LOW_RISK/heatmap"
n_top_patients = 5

feature_files = glob.glob(feature_name_pattern)
image_files = glob.glob(image_name_pattern)
patient_id = 'CIR-1429'
patient_feature_files = [file for file in feature_files if patient_id in file]
patient_image_files = [image for image in image_files if patient_id in image]

patient_data_files = []

model = ViT(num_classes=1, input_dim=1024)
model.load_state_dict(torch.load(learner_path, map_location=torch.device('cpu')))
#model.cuda().eval()
model.eval()

for file in patient_feature_files:
    patient_id = os.path.basename(file).split("_")[0]  # Extract patient ID from the feature file path
    matching_image_files = [
        image for image in patient_image_files if patient_id in image
        ]
    if matching_image_files:
        patient_data_files.append((file, matching_image_files[0]))
    
for file, image in patient_data_files:
    f_hname = os.path.basename(file)
    print(f"Processing {f_hname}...")
    if os.path.exists(output_folder + f"/{f_hname}_toptiles_layer_0.csv"):
        print("Exists. Skipping...")
        continue
    f = h5py.File(file)
    coords = f["coords"][:]
    xs = np.sort(np.unique(coords[:, 0]))
    stride = np.min(xs[1:] - xs[:-1])
    feats = torch.tensor(f["feats"][:]).float()
    
    feats.requires_grad = True
    output = model(feats.unsqueeze(0))
    output.backward()
    
    gradcam = (feats.grad * feats).mean(dim=-1).abs()

    scores = model(feats.unsqueeze(-2)).squeeze(-1)
    scores_min = scores.min(0, keepdim=True)[0]
    scores_max = scores.max(0, keepdim=True)[0]
    scores_normalised = (scores - scores_min)/(scores_max - scores_min)
    range_width = 2 # newMax - newMin = 1-(-1)
    new_scores = (scores_normalised * range_width) + (-1)
    weighted_gradcam = gradcam * new_scores
    n_top_tiles = 8
    top_gradcam_score_indices = weighted_gradcam.topk(n_top_tiles).indices
        
    img = vals_to_im(weighted_gradcam,coords//stride)
    extreme = abs(img).max()
    fig, ax = plt.subplots()
    cax = ax.imshow(img, alpha=np.float32(img != 0), cmap="RdBu_r", vmin=-(extreme), vmax=extreme)
    #for i in range(1,6):
    top_tile=coords[weighted_gradcam.topk(3).indices.cpu().numpy()]
    top_tile = top_tile[-1]
    rect = Rectangle(((top_tile//stride)[0], (top_tile//stride)[1]), -1, -1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.colorbar(cax)
    plt.show()
    fig.savefig(output_folder + f"/{f_hname}_marked_3_attention_map.png",dpi=300,)
