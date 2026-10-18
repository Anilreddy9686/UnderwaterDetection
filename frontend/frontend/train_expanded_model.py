from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXPANDED_WEIGHTS = BASE_DIR / "best_model_expanded.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a separate expanded underwater detector.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Use cpu, or set a CUDA device such as 0 when available.")
    args = parser.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / "Ultralytics"))
    from ultralytics import YOLO

    data_yaml = BASE_DIR / "training_data" / "expanded" / "data.yaml"
    starting_weights = BASE_DIR / "best_model+CSEA.pt"
    if not data_yaml.is_file():
        raise FileNotFoundError("Expanded dataset is missing. Run Update_Underwater_Model.bat first.")
    if not starting_weights.is_file():
        raise FileNotFoundError(f"Original model not found: {starting_weights}")

    print(f"Starting from the existing checkpoint: {starting_weights}")
    print("The original checkpoint will not be overwritten.")
    model = YOLO(str(starting_weights))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=8,
        cache=False,
        project=str(BASE_DIR / "runs" / "detect"),
        name="underwater_expanded",
        exist_ok=True,
        pretrained=True,
    )

    best_weights = BASE_DIR / "runs" / "detect" / "underwater_expanded" / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training completed without best weights at {best_weights}")
    shutil.copy2(best_weights, EXPANDED_WEIGHTS)
    print(f"Expanded model saved to: {EXPANDED_WEIGHTS}")
    print("This dataset does not contain labeled rocks, so the model will not detect rocks yet.")


if __name__ == "__main__":
    main()

