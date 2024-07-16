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
        #im[y:(y+stride), x:(x+stride)] = score.cpu().detach().numpy()
    return im

def save_tiles(
    image_path,
    top_tile_coords: npt.NDArray,
    #bottom_tile_coords: npt.NDArray,
    top_scores: npt.NDArray[np.float_],
    #bottom_scores: npt.NDArray[np.float_],
    output_folder,
    stride: float,
):
    assert len(top_tile_coords) == len(top_scores)
    #assert len(bottom_tile_coords) == len(bottom_scores)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    Image.MAX_IMAGE_PIXELS = None
    slide = openslide.open_slide(image_path)
    slide_mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    assert stride == 224, "probably not a problem, but better check"
    scaling_factor = (  # D:
        1  # unit: omar pixels
        / stride  # unit: tile index from the top-left corner
        * 256  # unit: um of the top left corner of the tile from the top left corner of the WSI
        / slide_mpp  # unit: WSI pixel
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    scaled_coordinates_top_tile = np.int32(top_tile_coords * scaling_factor)
    #scaled_coordinates_bottom_tile = np.int32(bottom_tile_coords * scaling_factor)

    #FOR TOP TILES
    for coords, score in zip(scaled_coordinates_top_tile, top_scores, strict=True):
        output_path = (
            output_folder
            / f"{image_name}_toptiles_{score:1.2e}_{coords[0]}_{coords[1]}.jpg"
        )
        tile = slide.read_region(
            (coords[0], coords[1]),
            0,
            # unit: wsi pixels
            (np.uint(stride * scaling_factor), np.uint(stride * scaling_factor)),
        ).convert("RGB")
        tile.save(output_path)

    #FOR BOTTOM TILES
    # for coords, score in zip(scaled_coordinates_bottom_tile, bottom_scores, strict=True):
    #     output_path = (
    #         output_folder
    #         / f"{image_name}_bottomtiles_{score}_{coords[0]}_{coords[1]}.jpg"
    #     )
    #     tile = slide.read_region(
    #         (coords[0], coords[1]),
    #         0,
    #         # unit: wsi pixels
    #         (np.uint(stride * scaling_factor), np.uint(stride * scaling_factor)),
    #     ).convert("RGB")
    #     tile.save(output_path)

learner_path = "/mnt/bulk-io/vidhya/karishma/DACHS/TFMS-UNI-BATCH64-404020/lr_1e-05_l1_0.001_l2_0.001_best_model.pth"
feature_name_pattern = "/mnt/bulk-io/vidhya/karishma/CIRCULATE/UNI-FEATURES/E2E_macenko_uni_old/*.h5"
image_name_pattern = "/mnt/bulk-io/vidhya/karishma/CIRCULATE/IMGS/*.ndpi"
scores_csv = "/mnt/bulk-io/vidhya/karishma/DACHS/TFMS-UNI-BATCH64-404020/CIRCULATE_score.csv"
output_folder = "/home/vidhya/marugoto/marugoto/survival/MID_RISK/"
n_top_patients = 5

feature_files = glob.glob(feature_name_pattern)
image_files = glob.glob(image_name_pattern)
patient_id = 'CIR-0645'
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
    print(top_gradcam_score_indices)
        
    img = vals_to_im(weighted_gradcam,coords//stride)
    extreme = abs(img).max()
    fig, ax = plt.subplots()
    cax = ax.imshow(img, alpha=np.float32(img != 0), cmap="RdBu_r", vmin=-(extreme), vmax=extreme)
    fig.colorbar(cax)
    plt.show()
    fig.savefig(output_folder + f"/{f_hname}_attention_map.png",dpi=300,)

    save_tiles(image,top_tile_coords=coords[top_gradcam_score_indices.numpy()],
               top_scores=weighted_gradcam[top_gradcam_score_indices].detach().cpu().numpy(),
               output_folder=output_folder,
               stride=stride)
