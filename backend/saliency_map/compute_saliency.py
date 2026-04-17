import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from PIL import Image

from read_data import ISICDataSet, ChestXrayDataSet, TBX11kDataSet, VINDRDataSet
from model import ConvNeXtV2
from explanations import SBSMBatch, SimCAM


def rank_retrieval(dists, labels, topk=1):
    """Find top-k closest embeddings for each query."""
    dists_copy = dists.copy()
    np.fill_diagonal(dists_copy, np.nan)

    idx = np.argsort(dists_copy, axis=1)[:, :topk]
    pred = labels[idx]
    return pred, idx


class ImageListDataSet(Dataset):
    def __init__(self, image_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name

    def __len__(self):
        return len(self.image_list)


def process(explainer, loader, device, args, gpu_count):
    if args.self_saliency:
        dataset = ImageListDataSet(
            image_dir="",
            image_list=loader.dataset.image_names,
            transform=loader.dataset.transform,
        )
        self_loader = DataLoader(
            dataset,
            batch_size=args.eval_batch_size * gpu_count,
            num_workers=args.workers,
        )

        for data in self_loader:
            samples, paths = data[0].to(device), data[1]
            if args.explainer == "sbsm":
                salmaps = explainer(samples)
            else:
                salmaps = explainer(samples, samples)

            salmaps = salmaps.cpu().numpy()

            os.makedirs(args.save_dir, exist_ok=True)
            for s, p in zip(salmaps, paths):
                np.save(os.path.join(args.save_dir, p.split("/")[-1]), s)
    else:
        results = np.load(args.results, allow_pickle=True)

        if "classification_results" in results:
            classification_data = results["classification_results"].item()
            dists = classification_data.get("dists")
            labels = classification_data.get("labels")
        else:
            dists = results["dists"]
            labels = results["labels"]

        _, idx = rank_retrieval(dists, labels, topk=args.topk)
        image_list = loader.dataset.image_names

        for img, ind in zip(image_list, idx):
            x_q = loader.dataset.transform(Image.open(img)).unsqueeze(0).to(device)
            x_q = torch.cat([x_q] * gpu_count)

            retrieved_images = [image_list[i] for i in ind]
            retrieved_dataset = ImageListDataSet(
                image_dir="",
                image_list=retrieved_images,
                transform=loader.dataset.transform,
            )
            retrieved_loader = DataLoader(
                retrieved_dataset,
                batch_size=args.eval_batch_size * gpu_count,
                num_workers=args.workers,
            )

            for data in retrieved_loader:
                samples, paths = data[0].to(device), data[1]
                salmaps = explainer(x_q, samples)
                salmaps = salmaps.cpu().numpy()

                base_path = os.path.join(args.save_dir, img.split("/")[-1])
                os.makedirs(base_path, exist_ok=True)

                for s, p in zip(reversed(salmaps), reversed(paths)):
                    np.save(os.path.join(base_path, p.split("/")[-1]), s)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_count = max(1, torch.cuda.device_count())

    model = ConvNeXtV2()

    if args.resume and os.path.isfile(args.resume):
        print("=> loading checkpoint")
        checkpoint = torch.load(args.resume)
        if "state-dict" in checkpoint:
            checkpoint = checkpoint["state-dict"]
        model.load_state_dict(checkpoint, strict=False)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    model.eval()

    if args.explainer == "sbsm":
        explainer = SBSMBatch(model, input_size=(384, 384), gpu_batch=args.gpu_batch)
        maskspath = "masks.npy"
        if not os.path.isfile(maskspath):
            explainer.generate_masks(window_size=24, stride=5, savepath=maskspath)
        else:
            explainer.load_masks(maskspath)
            print("Masks are loaded.")
    elif args.explainer == "simcam":
        backbone = model.convnext
        target_layer = backbone.stages[3].blocks[2]
        explainer = SimCAM(model=backbone, target_layer=target_layer, fc=None)
    else:
        raise NotImplementedError(
            "Explainer not supported for ConvNeXtV2. Use: sbsm or simcam"
        )

    explainer = explainer.to(device)
    explainer.eval()

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if args.dataset == "covid":
        test_dataset = ChestXrayDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=test_transform,
        )
    elif args.dataset == "isic":
        test_dataset = ISICDataSet(
            data_dir=args.test_dataset_dir,
            image_list_file=args.test_image_list,
            mask_dir=args.mask_dir,
            transform=test_transform,
        )
    elif args.dataset == "tbx11k":
        test_dataset = TBX11kDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=test_transform,
        )
    elif args.dataset == "vindr":
        test_dataset = VINDRDataSet(
            data_dir=args.test_dataset_dir,
            csv_file=args.test_image_list,
            transform=test_transform,
        )
    else:
        raise NotImplementedError("Dataset not supported!")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size * gpu_count,
        shuffle=False,
        num_workers=args.workers,
    )

    print("Evaluating ConvNeXtV2 saliency...")
    with torch.set_grad_enabled(args.explainer != "sbsm"):
        process(explainer, test_loader, device, args, gpu_count)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch Saliency Evaluation (ConvNeXtV2 only)"
    )

    parser.add_argument(
        "--dataset",
        default="covid",
        help="Dataset to use (covid, isic, tbx11k, or vindr)",
    )
    parser.add_argument(
        "--test-dataset-dir",
        default="/data/brian.hu/COVID/data/test",
        help="Test dataset directory path",
    )
    parser.add_argument(
        "--test-image-list", default="./test_COVIDx4.txt", help="Test image list"
    )
    parser.add_argument(
        "--mask-dir", default=None, help="Segmentation masks path (if used)"
    )
    parser.add_argument("--results", default=None, help="Results file to load")
    parser.add_argument(
        "--explainer",
        default="sbsm",
        choices=["sbsm", "simcam"],
        help="Explanation type (sbsm or simcam)",
    )
    parser.add_argument(
        "--self-saliency", action="store_true", help="Compute self-similarity saliency"
    )
    parser.add_argument("--eval-batch-size", default=1, type=int)
    parser.add_argument(
        "--gpu-batch",
        default=250,
        type=int,
        help="Internal batch size (only used for sbsm)",
    )
    parser.add_argument(
        "--topk", default=5, type=int, help="Number of top-k images to compute saliency"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save-dir", default="./saliency", help="Result save directory"
    )
    parser.add_argument("--resume", default="", help="Resume from checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
