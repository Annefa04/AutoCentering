
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT_DIR
    / "runs"
    / "obb"
    / "cable_476_v1"
    / "weights"
    / "best.pt"
)

INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output" / "model2_detection"

CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    if not image_paths:
        raise RuntimeError(f"No images found in {INPUT_DIR}")

    detected = 0
    missed = 0

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Cannot read: {image_path.name}")
            continue

        results = model.predict(
            source=image,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]
        output = image.copy()

        if result.obb is not None and len(result.obb) > 0:
            obb = result.obb
            boxes = obb.xywhr.cpu().numpy()
            scores = obb.conf.cpu().numpy()
            classes = obb.cls.cpu().numpy().astype(int)

            candidates = [
                i for i, cls_id in enumerate(classes)
                if cls_id == 0
            ]

            if candidates:
                best_index = max(
                    candidates,
                    key=lambda i: scores[i],
                )

                # Draw the predicted rotated box.
                polygon = cv2.boxPoints(
                    (
                        (
                            float(boxes[best_index][0]),
                            float(boxes[best_index][1]),
                        ),
                        (
                            float(boxes[best_index][2]),
                            float(boxes[best_index][3]),
                        ),
                        float(boxes[best_index][4]) * 180.0 / 3.141592653589793,
                    )
                ).astype("int32")

                cv2.polylines(
                    output,
                    [polygon],
                    True,
                    (255, 0, 0),
                    2,
                )

                confidence = float(scores[best_index])

                cv2.putText(
                    output,
                    f"Model 2: {confidence:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                detected += 1
            else:
                cv2.putText(
                    output,
                    "No sample detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                missed += 1
        else:
            cv2.putText(
                output,
                "No sample detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            missed += 1

        output_path = OUTPUT_DIR / f"{image_path.stem}_result.png"
        cv2.imwrite(str(output_path), output)

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} processed"
        )

    print("\nTesting complete.")
    print(f"Images detected: {detected}")
    print(f"Images without detection: {missed}")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
