
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Project paths
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
OUTPUT_DIR = ROOT_DIR / "output" / "auto_center"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CONFIDENCE = 0.25
IMAGE_SIZE = 640


def process_image(image_path, model):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Cannot open image: {image_path.name}")
        return

    height, width = image.shape[:2]

    # Centre of the image
    image_cx = width / 2
    image_cy = height / 2

    result = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    display = image.copy()

    # Draw image centre (blue)
    cv2.drawMarker(
        display,
        (round(image_cx), round(image_cy)),
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2
    )

    if result.obb is None or len(result.obb) == 0:
        print(f"\nImage: {image_path.name}")
        print("No component detected.")
        cv2.putText(
            display, "NO COMPONENT DETECTED",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 255), 2
        )
    else:
        # Select the detection with the highest confidence
        confidences = result.obb.conf.cpu().numpy()
        best_index = int(np.argmax(confidences))

        corners = result.obb.xyxyxyxy[best_index].cpu().numpy()
        confidence = float(confidences[best_index])

        # Centre of the four OBB corners
        component_cx = float(corners[:, 0].mean())
        component_cy = float(corners[:, 1].mean())

        # Positive dx means component is right of image centre.
        # Positive dy means component is below image centre.
        dx = component_cx - image_cx
        dy = component_cy - image_cy
        distance = float(np.hypot(dx, dy))

        # Draw rotated bounding box (green)
        polygon = np.round(corners).astype(np.int32)
        cv2.polylines(
            display, [polygon], True, (0, 255, 0), 2
        )

        # Draw component centre (red)
        cv2.circle(
            display,
            (round(component_cx), round(component_cy)),
            7, (0, 0, 255), -1
        )

        # Connect image centre and component centre
        cv2.line(
            display,
            (round(image_cx), round(image_cy)),
            (round(component_cx), round(component_cy)),
            (0, 255, 255), 2
        )

        info = [
            f"Confidence: {confidence:.3f}",
            f"Image center: ({image_cx:.1f}, {image_cy:.1f})",
            f"Object center: ({component_cx:.1f}, {component_cy:.1f})",
            f"Offset: dx={dx:+.1f}px, dy={dy:+.1f}px",
            f"Distance from center: {distance:.1f}px"
        ]

        for i, text in enumerate(info):
            cv2.putText(
                display, text, (20, 35 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA
            )

        print(f"\nImage: {image_path.name}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Image centre: ({image_cx:.1f}, {image_cy:.1f})")
        print(f"Component centre: ({component_cx:.1f}, {component_cy:.1f})")
        print(f"Offset: dx={dx:+.1f}px, dy={dy:+.1f}px")
        print(f"Distance from centre: {distance:.1f}px")

    output_path = OUTPUT_DIR / f"{image_path.stem}_centered_test.png"
    cv2.imwrite(str(output_path), display)
    print(f"Saved result: {output_path}")


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found:\n{MODEL_PATH}")
        return

    if not INPUT_DIR.exists():
        print(f"ERROR: Input folder not found:\n{INPUT_DIR}")
        return

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        print(f"No supported images found in: {INPUT_DIR}")
        print("Put at least one X-ray image in the input folder.")
        return

    print(f"Model: {MODEL_PATH}")
    print(f"Images found: {len(images)}")

    model = YOLO(str(MODEL_PATH))

    for image_path in images:
        process_image(image_path, model)

    print("\nAuto-centering calculation test finished.")
    print(f"Open the output folder:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
