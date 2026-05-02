import argparse
from pathlib import Path

import torch
import torch.nn as nn

import sys


CUR_PATH = Path(__file__).resolve().parent.parent
YOLO_DIR = CUR_PATH / "yolo5"
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

import models  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from utils.activations import Hardswish, SiLU  # noqa: E402
from utils.general import check_img_size, set_logging  # noqa: E402


def convert_one(weights_path: Path, img_size=640, batch_size=1):
    weights_path = weights_path.resolve()
    onnx_path = weights_path.with_suffix(".onnx")
    if onnx_path.exists():
        print(f"skip_exists={onnx_path}")
        return onnx_path

    model = attempt_load(str(weights_path), device=torch.device("cpu"))
    gs = int(max(model.stride))
    img_size = [check_img_size(img_size, gs), check_img_size(img_size, gs)]
    img = torch.zeros(batch_size, 3, *img_size)

    for _, module in model.named_modules():
        module._non_persistent_buffers_set = set()
        if isinstance(module, models.common.Conv):
            if isinstance(module.act, nn.Hardswish):
                module.act = Hardswish()
            elif isinstance(module.act, nn.SiLU):
                module.act = SiLU()
    model.model[-1].export = True
    model(img)

    torch.onnx.export(
        model,
        img,
        str(onnx_path),
        verbose=False,
        opset_version=12,
        input_names=["images"],
        output_names=["output"],
        dynamo=False,
    )
    print(f"exported={onnx_path}")
    return onnx_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=CUR_PATH / "weights",
        help="Directory containing .pt weights",
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--glob",
        default="stage*_fold*.pt",
        help="Glob pattern for weight files",
    )
    args = parser.parse_args()

    set_logging()
    weights = sorted(args.weights_dir.glob(args.glob))
    if not weights:
        raise FileNotFoundError(f"No weights matched {args.glob} in {args.weights_dir}")

    for weights_path in weights:
        if weights_path.name.endswith(".torchscript.pt"):
            continue
        convert_one(weights_path, img_size=args.img_size, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
