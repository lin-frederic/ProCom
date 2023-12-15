"""Run the main experiment"""

import os
import json
import torch
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import wandb
from tools import PadAndResize, ResizeModulo
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from classif.ncm import NCM
from models.maskblocks import Identity, Lost, DeepSpectralMethods, SAM, combine_masks
from augment.augmentations import crop_mask
from dataset import DatasetBuilder


def baseline(cfg):
    """Run the baseline experiment"""
    sampler = DatasetBuilder(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s", use_v2=False)
    model.to(device)
    model.eval()

    resize = T.Resize((224, 224))

    transforms = T.Compose(
        [
            PadAndResize(224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # imagenet mean and std
        ]
    )

    identity = Identity()
    L_acc = []
    ncm = NCM()
    dataset = cfg.dataset
    pbar = tqdm(range(cfg.n_runs), desc="Runs")

    for episode_idx in pbar:

        # new sample for each run
        episode = sampler(
            seed_classes=episode_idx, seed_images=episode_idx
        )  # episode is (dataset, classe, support/query, image_path)

        sample = episode[dataset]

        support_images, temp_support_labels, query_images, temp_query_labels = sample

        # strategy: take the identity mask + the top k (config) sam masks

        support_augmented_imgs = []
        support_labels = []

        for i, img_path in enumerate(support_images):
            img = resize(Image.open(img_path).convert("RGB"))
            masks_id = identity(img)
            masks = masks_id
            support_augmented_imgs += [
                crop_mask(img, mask["segmentation"]) for mask in masks
            ]
            labels = [(temp_support_labels[i], i) for j in range(len(masks))]
            support_labels += labels

        query_augmented_imgs = []
        query_labels = []
        for i, img_path in enumerate(query_images):
            img = resize(Image.open(img_path).convert("RGB"))
            masks_id = identity(img)
            masks = masks_id
            query_augmented_imgs += [
                crop_mask(img, mask["segmentation"]) for mask in masks
            ]
            labels = [(temp_query_labels[i], i) for j in range(len(masks))]
            query_labels += labels

        support_augmented_imgs = [
            transforms(img).to(device) for img in support_augmented_imgs
        ]
        query_augmented_imgs = [
            transforms(img).to(device) for img in query_augmented_imgs
        ]

        support_tensor = torch.zeros(
            (len(support_augmented_imgs), 384)
        )  # size of the feature vector
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        bs = cfg.batch_size

        with torch.inference_mode():
            for i in range(0, len(support_augmented_imgs), bs):
                inputs = torch.stack(support_augmented_imgs[i: i + bs])
                outputs = model(inputs)
                support_tensor[i: i + bs] = outputs

            for i in range(0, len(query_augmented_imgs), bs):
                inputs = torch.stack(query_augmented_imgs[i: i + bs])
                outputs = model(inputs)
                query_tensor[i: i + bs] = outputs

        acc = ncm(support_tensor, query_tensor, support_labels, query_labels)

        L_acc.append(acc)

        if cfg.wandb:
            wandb.log(
                {
                    "running_accuracy": acc,
                    "average_accuracy": np.mean(L_acc),
                }
            )

        pbar.set_description(
            f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

    print(
        "Average accuracy: ", round(
            np.mean(L_acc), 2), "std: ", round(
            np.std(L_acc), 2))
    print("All accuracies: ", np.round(L_acc, 2))


def main(cfg):
    """Run the main experiment"""
    sampler = DatasetBuilder(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s", use_v2=False)
    model.to(device)
    model.eval()

    resize = T.Resize((224, 224))

    transforms = T.Compose(
        [
            ResizeModulo(patch_size=16, target_size=224),
            T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # imagenet mean and std
        ]
    )

    lost_deg_seed = Lost(alpha=0.95, k=100, model=model)
    lost_atn_seed = Lost(alpha=0.05, k=100, model=model)
    spectral = DeepSpectralMethods(
        model=model, n_eigenvectors=15, lambda_color=5)
    sam = SAM()

    L_acc = []
    ncm = NCM()
    dataset = cfg.dataset

    pbar = tqdm(range(cfg.n_runs), desc="Runs")
    if not os.path.exists(f"{cfg.sam_cache}/{dataset}/cache.json"):
        # create the cache
        json.dump(
            {},
            open(
                f"{cfg.sam_cache}/{dataset}/cache.json",
                "w",
                encoding="utf-8"))
    with open(f"{cfg.sam_cache}/{dataset}/cache.json", "r", encoding="utf-8") as f:
        sam_cache = json.load(f)

    for episode_idx in pbar:

        # new sample for each run
        episode = sampler(seed_classes=episode_idx, seed_images=episode_idx)
        # episode is (dataset, classe, support/query, image_path)

        sample = episode[dataset]

        support_images, temp_support_labels, query_images, temp_query_labels = sample
        support_augmented_imgs = []
        support_labels = []
        for i, img_path in enumerate(support_images):
            img = resize(Image.open(img_path).convert("RGB"))
            # concatenate the output of the two lost blocks
            mask_lost = lost_deg_seed(img) + lost_atn_seed(img)
            # average area of the lost masks
            approx_area = np.mean([mask["area"] for mask in mask_lost])

            # filter out masks that are too small
            if img_path not in sam_cache:
                mask_sam = sam(img)
                mask_sam_f = [
                    mask for mask in mask_sam if mask["area"] > 0.4 *
                    approx_area]
                # keep the same magnitude of area as the lost masks
                if len(mask_sam_f) > 2:
                    mask_sam = mask_sam_f
                img_name = img_path.split("/")[-1]
                img_name = img_name.split(".")[0]
                masks = [mask_sam[j]["segmentation"]
                         for j in range(len(mask_sam))]
                areas = [mask_sam[j]["area"] for j in range(len(mask_sam))]
                masks = np.array(masks)
                areas = np.array(areas)
                cache_name = f"{cfg.sam_cache}/{dataset}/{img_name}.npz"
                np.savez_compressed(cache_name, masks=masks, areas=areas)
                sam_cache[img_path] = cache_name
                with open(f"{cfg.sam_cache}/{dataset}/cache.json", "w") as f:
                    json.dump(sam_cache, f)
            else:
                cache = np.load(sam_cache[img_path])
                masks = cache["masks"]
                areas = cache["areas"]
                mask_sam = [
                    {"area": areas[j], "segmentation": masks[j]}
                    for j in range(len(masks))
                ]

            mask_spectral = spectral(img)

            masks_sam, masks_spectral, _ = combine_masks(
                mask_sam, mask_spectral, mask_lost, norm=True, postprocess=True
            )
            masks = masks_sam[: cfg.top_k_masks] + \
                masks_spectral[: cfg.top_k_masks]
            support_augmented_imgs += [
                crop_mask(img, mask["segmentation"]) for mask in masks
            ]
            labels = [(temp_support_labels[i], i) for j in range(len(masks))]
            support_labels += labels

        query_augmented_imgs = []
        query_labels = []

        for i, img_path in enumerate(query_images):
            img = resize(Image.open(img_path).convert("RGB"))
            mask_lost = lost_deg_seed(img) + lost_atn_seed(img)
            approx_area = np.mean([mask["area"] for mask in mask_lost])

            # filter out masks that are too small
            if img_path not in sam_cache:
                mask_sam = sam(img)
                mask_sam_f = [
                    mask for mask in mask_sam if mask["area"] > 0.4 *
                    approx_area]
                # keep the same magnitude of area as the lost masks
                if len(mask_sam_f) > 2:
                    mask_sam = mask_sam_f
                img_name = img_path.split("/")[-1]
                img_name = img_name.split(".")[0]
                masks = [mask_sam[j]["segmentation"]
                         for j in range(len(mask_sam))]
                areas = [mask_sam[j]["area"] for j in range(len(mask_sam))]
                masks = np.array(masks)
                areas = np.array(areas)
                cache_name = f"{cfg.sam_cache}/{img_name}.npz"
                np.savez_compressed(cache_name, masks=masks, areas=areas)
                sam_cache[img_path] = cache_name
                with open(cfg.sam_cache + "/cache.json", "w", encoding="utf-8") as f:
                    json.dump(sam_cache, f)
            else:
                cache = np.load(sam_cache[img_path])
                masks = cache["masks"]
                areas = cache["areas"]
                mask_sam = [
                    {"area": areas[j], "segmentation": masks[j]}
                    for j in range(len(masks))
                ]

            mask_spectral = spectral(img)

            masks_sam, masks_spectral, _ = combine_masks(
                mask_sam, mask_spectral, mask_lost, norm=True, postprocess=True
            )
            masks = masks_sam[: cfg.top_k_masks] + \
                masks_spectral[: cfg.top_k_masks]
            query_augmented_imgs += [
                crop_mask(img, mask["segmentation"]) for mask in masks
            ]

            labels = [(temp_query_labels[i], i) for j in range(len(masks))]
            query_labels += labels

        support_augmented_imgs = [
            transforms(img).to(device) for img in support_augmented_imgs
        ]
        query_augmented_imgs = [
            transforms(img).to(device) for img in query_augmented_imgs
        ]

        # size of the feature vector, WARNING: this is hardcoded for the model
        # used
        support_tensor = torch.zeros((len(support_augmented_imgs), 384))
        query_tensor = torch.zeros((len(query_augmented_imgs), 384))

        # bs = cfg.batch_size

        with torch.inference_mode():
            for i in range(len(support_augmented_imgs)):
                inputs = support_augmented_imgs[i].unsqueeze(0)
                """to_display = inputs.squeeze(0).permute(1,2,0).cpu().numpy()
                to_display = (to_display - np.min(to_display))/(np.max(to_display) - np.min(to_display))*255
                to_display = to_display.astype(np.uint8)
                Image.fromarray(to_display).save(f"results/{i}_{uuid4()}.png")"""
                outputs = model(inputs).squeeze(0)
                support_tensor[i] = outputs

            for i in range(len(query_augmented_imgs)):
                inputs = query_augmented_imgs[i].unsqueeze(0)
                outputs = model(inputs).squeeze(0)
                query_tensor[i] = outputs

        acc = ncm(support_tensor, query_tensor, support_labels, query_labels)

        L_acc.append(acc)
        pbar.set_description(
            f"Last: {round(acc,2)}, avg: {round(np.mean(L_acc),2)}")

        if cfg.wandb:
            wandb.log(
                {
                    "running_accuracy": acc,
                    "average_accuracy": np.mean(L_acc),
                }
            )

    print(
        "Average accuracy: ", round(
            np.mean(L_acc), 2), "std: ", round(
            np.std(L_acc), 2))
    print("All accuracies: ", np.round(L_acc, 2))


if __name__ == "__main__":

    print("Config:", cfg.sampler)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="baseline",
        help="baseline, nosam, main")
    parser.add_argument("--wandb", "-w", action="store_true", help="use wandb")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="imagenet",
        help="imagenet, cub, caltech, food, cifarfs, fungi, flowers, pets",
    )
    args = parser.parse_args()

    cfg["type"] = args.type
    cfg["dataset"] = args.dataset

    if args.wandb:
        wandb.login()
        wandb.init(project="procom-transformers", entity="procom")
        cfg["wandb"] = True
        wandb.config.update(cfg)

    if args.type == "baseline":
        baseline(cfg)

    elif args.type == "main":
        main(cfg)
