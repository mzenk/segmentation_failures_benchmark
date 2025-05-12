import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import natsort
import nibabel as nib
import numpy as np
import pandas as pd
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from PIL import Image
from scipy.io import loadmat

from segmentation_failures.utils.io import load_json, save_json

# # from their reference code:
# import numpy as np
# import os
# import natsort
# from PIL import Image
# from skimage import transform

# bmpfile_path='/home/Data/OCTA-500/OCTA_6mm/OCT'
# npzfile_path='/home/Data/OCTA-500/OCTA_6mm/OCT_npy'
# ctlist = os.listdir(bmpfile_path)
# ctlist = natsort.natsorted(ctlist)
# for ct in ctlist:
#     data = []
#     bscanlist=os.listdir(os.path.join(bmpfile_path,ct))
#     bscanlist = natsort.natsorted(bscanlist)
#     for bscan in bscanlist:
#         data.append(np.array(Image.open(os.path.join(bmpfile_path,ct,bscan)).resize((400,128))))
#     np.save(os.path.join(npzfile_path,ct),data)


def process_case(
    bmp_folder,
    label_path,
    output_img_path,
    output_label_path,
    fov_mm,
    crop: int = 0,
    dry_run=False,
):
    file_ending = ".".join(str(output_img_path).split(".")[1:])
    if file_ending == "":
        output_img_path += ".nii.gz"
        output_label_path += ".nii.gz"
    elif file_ending != "nii.gz":
        raise ValueError("Output image path must be a .nii.gz file. Got:" + file_ending)
    # FOV_mm: Field of view in mm, dimensions: (bscan_1, bscan_2, across_bscans)
    # Step 1: Get list of BMP files
    bmp_files = natsort.natsorted(
        [f for f in Path(bmp_folder).iterdir() if f.name.endswith(".bmp")]
    )
    # print(bmp_files)
    # Step 2: Load each BMP file and stack them into a 3D array
    slices = []
    for bmp_file in bmp_files:
        # Load the BMP image as a grayscale image
        img = Image.open(bmp_file)
        img_arr = np.array(img).T
        if img_arr.dtype != np.uint8:
            print("WARNING: dtype", img_arr.dtype, "for", bmp_file)
        slices.append(img_arr)

    # Convert the list of slices to a 3D numpy array
    volume = (
        np.stack(slices, axis=0).astype(np.float32) / 255.0
    )  # Normalize pixel values to [0, 1]
    fov_mm = (float(fov_mm), float(fov_mm), 2.0)
    spacings = [fov_mm[i] / (volume.shape[i] - 1) for i in range(3)]
    # process and save label
    label_data = loadmat(label_path)["Layer"]
    if label_data.min() < 0.01 * volume.shape[2] or label_data.max() > 0.99 * volume.shape[2]:
        print(f"WARNING: label data in {label_path} is unusual. Check manually")

    label_volume = np.zeros_like(volume, dtype=np.uint8)
    for label_idx in range(5):
        for i in range(label_volume.shape[0]):
            for j in range(label_volume.shape[1]):
                label_volume[
                    i,
                    j,
                    label_data[label_idx, i, j] : label_data[label_idx + 1, i, j],
                ] = (
                    1 + label_idx
                )
    # Optional: crop the volume above/below the retinal layer foreground
    if crop > 0:
        # same from below
        min_idx = np.nonzero(label_volume.sum(axis=(0, 1)))[0].min()
        max_idx = np.nonzero(label_volume.sum(axis=(0, 1)))[0].max()
        crop_min_idx = max(0, min_idx - crop)
        crop_max_idx = min(max_idx + crop, volume.shape[2] - 1)
        volume = volume[:, :, crop_min_idx:crop_max_idx]
        label_volume = label_volume[:, :, crop_min_idx:crop_max_idx]
    # save niftis
    affine = np.diag(spacings + [1.0])
    affine[2, 2] *= -1
    nifti_img = nib.Nifti1Image(volume, affine=affine)
    nifti_label = nib.Nifti1Image(label_volume, affine=affine)
    if not dry_run:
        nib.save(nifti_img, output_img_path)
        nib.save(nifti_label, output_label_path)
        print(f"NIfTI files saved at:\n{output_img_path}\n{output_label_path}")
    stats = {
        "min": volume.min(),
        "max": volume.max(),
        "dtype": volume.dtype,
        "shape": volume.shape,
        "label_shape": label_volume.shape,
        "label_min": label_volume.min(),
        "label_max": label_volume.max(),
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_dir", type=str, help="Path to raw data directory")
    parser.add_argument("--exclude_3mm", action="store_true", help="Exclude 3mm OCTA scans")
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=40,
        help="Crop margin (pixels) above and below the retinal layer foreground.",
    )
    parser.add_argument(
        "--use_default_splits",
        action="store_true",
        help="Use default train/test split.",
    )
    args = parser.parse_args()
    source_dir = Path(args.raw_data_dir)
    if not source_dir.exists():
        raise FileNotFoundError("One of the specified directories does not exist")
    if args.exclude_3mm:
        TASK_NAME = "Dataset561_OCTA500_6mm_pathology_split"
    else:
        TASK_NAME = "Dataset560_OCTA500_pathology_split"
    print(f"Processing data for task {TASK_NAME}")

    nnunet_split = None
    if args.use_default_splits:
        default_split_path = (
            Path(__file__).resolve().parents[4]
            / "dataset_splits"
            / TASK_NAME
            / "splits_final.json"
        )
        nnunet_split = load_json(default_split_path)
    target_root_dir = Path(os.environ["nnUNet_raw"]) / TASK_NAME
    target_root_dir.mkdir(exist_ok=True)

    images_train_dir = target_root_dir / "imagesTr"
    images_test_dir = target_root_dir / "imagesTs"
    labels_train_dir = target_root_dir / "labelsTr"
    labels_test_dir = target_root_dir / "labelsTs"
    images_train_dir.mkdir()
    labels_train_dir.mkdir()
    images_test_dir.mkdir()
    labels_test_dir.mkdir()
    # get a list of all patients for each scanner and do the train/test split
    case_dict = {}
    all_cases = []
    metadata = pd.read_excel(source_dir / "OCTA_6mm" / "Text labels.xlsx")
    metadata["FOV"] = "6mm"
    for case_dir in (source_dir / "OCTA_6mm" / "OCT").iterdir():
        case_id = case_dir.name
        case_dict[case_id] = {
            "img_dir": case_dir,
            "label_path": source_dir / "Label" / "GT_Layers" / f"{case_id}.mat",
        }

    if not args.exclude_3mm:
        for case_dir in (source_dir / "OCTA_3mm" / "OCT").iterdir():
            case_id = case_dir.name
            case_dict[case_id] = {
                "img_dir": case_dir,
                "label_path": source_dir / "Label" / "GT_Layers" / f"{case_id}.mat",
            }
        tmp = pd.read_excel(source_dir / "OCTA_3mm" / "Text labels.xlsx")
        tmp["FOV"] = "3mm"
        metadata = pd.concat([metadata, tmp], ignore_index=True)
    metadata.set_index(metadata["ID"].astype(str), inplace=True)
    indistr_diseases = ["NORMAL", "AMD", "CNV"]
    if nnunet_split is not None:
        train_cases = []
        for fold in nnunet_split:
            train_cases.extend(fold["train"])
        test_cases = list(set(all_cases) - set(train_cases))
    else:
        rnd = np.random.default_rng(8903764)
        train_cases = []
        test_cases = []
        ind_test_fraction = 0.2
        ind_mask = metadata["Disease"].isin(indistr_diseases)
        ood_cases = metadata[~ind_mask].index.tolist()
        ind_test_cases = []
        for _, df in metadata[ind_mask].groupby(["FOV", "Disease"]):
            n = int(max(ind_test_fraction * len(df), 1))
            if len(df) <= 1:
                n = 0
            sampled_ids = df.sample(n, random_state=rnd).index.tolist()
            ind_test_cases.extend(sampled_ids)
        test_cases = ind_test_cases + ood_cases
        train_cases = list(set(metadata.index) - set(test_cases))
        print("=== Train set stats ===")
        stats = metadata.loc[train_cases, ["Disease", "FOV"]].groupby(["Disease", "FOV"]).size()
        stats = (
            pd.DataFrame({"count": stats})
            .reset_index()
            .pivot(index="Disease", columns="FOV", values="count")
        )
        print(stats)
        print("=== Test set stats ===")
        stats = metadata.loc[test_cases, ["Disease", "FOV"]].groupby(["Disease", "FOV"]).size()
        stats = (
            pd.DataFrame({"count": stats})
            .reset_index()
            .pivot(index="Disease", columns="FOV", values="count")
        )
        print(stats)

    # process all cases
    case_to_domain_map = {}
    results = []
    print(f"Processing {len(train_cases)} training cases and {len(test_cases)} test cases")
    with ProcessPoolExecutor(max_workers=4) as executor:
        for case_id, d in case_dict.items():
            if case_id in test_cases:
                img_out_dir = images_test_dir
                lab_out_dir = labels_test_dir
                case_to_domain_map[case_id] = metadata.loc[case_id, "Disease"]
            else:
                img_out_dir = images_train_dir
                lab_out_dir = labels_train_dir
            fov = metadata.at[case_id, "FOV"]
            if fov in ["3mm", "6mm"]:
                fov = float(fov[0])
            else:
                raise ValueError(f"Invalid FOV value: {fov}")
            results.append(
                executor.submit(
                    process_case,
                    d["img_dir"],
                    d["label_path"],
                    img_out_dir / f"{case_id}_0000.nii.gz",
                    lab_out_dir / f"{case_id}.nii.gz",
                    fov_mm=fov,
                    crop=args.crop_margin,
                )
            )
    img_shapes = []
    for r in results:
        stats = r.result()
        img_shapes.append(stats["shape"])
    img_shapes = np.array(img_shapes)
    print("=== Size statistics ===")
    overall_min = np.min(img_shapes, axis=0)
    overall_max = np.max(img_shapes, axis=0)
    median = np.median(img_shapes, axis=0)
    print(f"Overall min: {overall_min}, max: {overall_max}")
    print(f"Overall median: {median}")

    # save metadata etc
    save_json(case_to_domain_map, target_root_dir / "domain_mapping_00.json")
    label_dict = {
        "background": 0,
        "ILM-IPL": 1,
        "IPL-OPL": 2,
        "OPL-ISOS": 3,
        "ISOS-RPE": 4,
        "RPE-BM": 5,
    }
    metadata.to_csv(target_root_dir / "metadata.csv")
    generate_dataset_json(
        output_folder=str(target_root_dir),
        channel_names={0: "OCT"},
        labels=label_dict,
        num_training_cases=len(train_cases),
        file_ending=".nii.gz",
        dataset_name=TASK_NAME,
        dim=3,
    )


if __name__ == "__main__":
    main()
    # no need for special train-val splits here
