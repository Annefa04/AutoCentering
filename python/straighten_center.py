
from pathlib import Path
import math

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    r"C:\Users\R&D\XRayAutoCentering\runs\obb\runs\obb\xray_auto_centering_v1\weights\best.pt"
)

INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output" / "auto_orientation_centering"

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}

CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"

SAVE_PREVIEWS = True
MIN_ROTATION_DEGREES = 0.15


# ============================================================
# READ AND SAVE IMAGES
# ============================================================

def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)

        if data.size == 0:
            return None

        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    except Exception as exc:
        print(f"Image read error: {exc}")
        return None


def save_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)

    extension = path.suffix.lower() or ".png"
    success, encoded = cv2.imencode(extension, image)

    if not success:
        raise RuntimeError(f"Could not encode image: {path}")

    encoded.tofile(str(path))


# ============================================================
# YOLO OBB DETECTION
# ============================================================

def detect_component(model, image):
    results = model.predict(
        source=image,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        return None

    obb = getattr(results[0], "obb", None)

    if obb is None or len(obb) == 0:
        return None

    corners_all = obb.xyxyxyxy.cpu().numpy()
    confidences = obb.conf.cpu().numpy()
    xywhr = obb.xywhr.cpu().numpy()

    # Select the largest detected component.
    areas = xywhr[:, 2] * xywhr[:, 3]
    index = int(np.argmax(areas))

    corners = np.asarray(
        corners_all[index], dtype=np.float64
    ).reshape(4, 2)

    return {
        "corners": corners,
        "width": float(xywhr[index, 2]),
        "height": float(xywhr[index, 3]),
        "angle_rad": float(xywhr[index, 4]),
        "confidence": float(confidences[index]),
        "centre": corners.mean(axis=0),
    }


# ============================================================
# CALCULATE ROTATION
# ============================================================

def get_rotation_correction(detection):
    """
    Estimate orientation from the longest OBB edge.

    Returns:
        correction: angle passed to OpenCV
        original_angle: estimated edge angle
        explanation: diagnostic message

    Note:
    An undirected rectangle cannot tell us whether the object
    is upside down. Near-square boxes can have unstable angles.
    """

    corners = np.asarray(
        detection["corners"], dtype=np.float64
    )

    if corners.shape != (4, 2):
        return 0.0, 0.0, "Skipped: invalid OBB corners"

    if not np.isfinite(corners).all():
        return 0.0, 0.0, "Skipped: invalid coordinates"

    # Find the longest edge of the detected rectangle.
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)

    longest_index = int(np.argmax(lengths))
    edge = edges[longest_index]

    if lengths[longest_index] < 1e-6:
        return 0.0, 0.0, "Skipped: invalid edge length"

    angle = math.degrees(
        math.atan2(float(edge[1]), float(edge[0]))
    )

    # Rectangle orientation repeats every 180 degrees.
    normalized = (angle + 90.0) % 180.0 - 90.0

    # Snap to the nearest horizontal or vertical axis.
    target = round(normalized / 90.0) * 90.0

    # Image y-coordinates increase downwards.
    # For OpenCV, this sign straightens the detected edge.
    correction = normalized - target

    while correction > 90.0:
        correction -= 180.0

    while correction < -90.0:
        correction += 180.0

    short_side = max(
        min(detection["width"], detection["height"]),
        1e-6,
    )

    long_side = max(
        detection["width"],
        detection["height"],
    )

    aspect_ratio = long_side / short_side

    if aspect_ratio < 1.08:
        explanation = (
            f"Near-square box, aspect ratio={aspect_ratio:.2f}; "
            "angle may be unstable; using longest-edge estimate"
        )
    else:
        explanation = (
            f"Longest-edge angle={angle:.2f}; "
            f"normalised={normalized:.2f}; "
            f"target axis={target:.1f}"
        )

    if abs(correction) < MIN_ROTATION_DEGREES:
        correction = 0.0
        explanation += "; no rotation needed"

    return correction, angle, explanation


# ============================================================
# ROTATE IMAGE WITHOUT CROPPING
# ============================================================

def rotate_expand(image, angle_degrees):
    if abs(angle_degrees) < 1e-6:
        return image.copy()

    height, width = image.shape[:2]
    centre = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(
        centre, angle_degrees, 1.0
    )

    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])

    new_width = int(
        round(height * sine + width * cosine)
    )

    new_height = int(
        round(height * cosine + width * sine)
    )

    matrix[0, 2] += new_width / 2.0 - centre[0]
    matrix[1, 2] += new_height / 2.0 - centre[1]

    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


# ============================================================
# CENTRE DETECTED COMPONENT
# ============================================================

def center_component(image, detection):
    height, width = image.shape[:2]

    image_centre_x = width / 2.0
    image_centre_y = height / 2.0

    object_centre = np.asarray(
        detection["corners"], dtype=np.float64
    ).mean(axis=0)

    dx = image_centre_x - float(object_centre[0])
    dy = image_centre_y - float(object_centre[1])

    matrix = np.array(
        [
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
        ],
        dtype=np.float32,
    )

    centred = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return centred, dx, dy


# ============================================================
# CREATE PREVIEW
# ============================================================

def create_preview(image, detection):
    preview = image.copy()

    height, width = preview.shape[:2]

    corners = np.round(
        detection["corners"]
    ).astype(np.int32)

    object_centre = tuple(
        np.round(corners.mean(axis=0)).astype(int)
    )

    image_centre = (width // 2, height // 2)

    cv2.polylines(
        preview,
        [corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    cv2.circle(
        preview,
        object_centre,
        5,
        (0, 0, 255),
        -1,
    )

    cv2.drawMarker(
        preview,
        image_centre,
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=25,
        thickness=2,
    )

    cv2.putText(
        preview,
        f"Confidence: {detection['confidence']:.2f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(model, image_path):
    print(f"\nProcessing: {image_path.name}")

    image = read_image(image_path)

    if image is None:
        print("ERROR: Could not read image.")
        return False

    initial_detection = detect_component(model, image)

    if initial_detection is None:
        print("No component detected. Keeping original image.")
        corrected = image.copy()
        final_detection = None

    else:
        print(
            f"Confidence: "
            f"{initial_detection['confidence']:.3f}"
        )

        rotation, original_angle, explanation = (
            get_rotation_correction(initial_detection)
        )

        print(
            f"Detected OBB angle: "
            f"{original_angle:.2f} degrees"
        )

        print(
            f"Rotation correction: "
            f"{rotation:.2f} degrees"
        )

        print(f"Decision: {explanation}")

        corrected = rotate_expand(image, rotation)

        # Detect again after rotation.
        rotated_detection = detect_component(
            model, corrected
        )

        if rotated_detection is None:
            print(
                "WARNING: Re-detection failed. "
                "Saving rotated image without centering."
            )
            final_detection = None

        else:
            corrected, dx, dy = center_component(
                corrected,
                rotated_detection,
            )

            print(
                f"Center shift: dx={dx:.1f}, "
                f"dy={dy:.1f} pixels"
            )

            # Final detection for the preview.
            final_detection = detect_component(
                model, corrected
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_corrected.png"
    )

    save_image(output_path, corrected)

    print(f"Saved corrected image: {output_path}")

    if SAVE_PREVIEWS:
        if final_detection is not None:
            preview = create_preview(
                corrected,
                final_detection,
            )
        else:
            preview = corrected.copy()

        preview_path = (
            OUTPUT_DIR
            / f"{image_path.stem}_preview.png"
        )

        save_image(preview_path, preview)

        print(f"Saved preview: {preview_path}")

    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("X-RAY AUTO ORIENTATION AND CENTERING")
    print("=" * 60)

    print(f"Project folder: {ROOT_DIR}")
    print(f"Input folder:   {INPUT_DIR}")
    print(f"Output folder:  {OUTPUT_DIR}")
    print(f"Model path:     {MODEL_PATH}")

    if not MODEL_PATH.is_file():
        print("\nERROR: YOLO model not found.")
        print("Check MODEL_PATH at the top of this script.")
        return

    if not INPUT_DIR.is_dir():
        print("\nERROR: Input folder not found.")
        return

    image_paths = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print("\nNo supported images found in input folder.")
        return

    print(f"\nImages found: {len(image_paths)}")
    print("Loading YOLO model...")

    try:
        model = YOLO(str(MODEL_PATH))
    except Exception as exc:
        print(f"ERROR loading model: {exc}")
        return

    print("Model loaded.")

    succeeded = 0
    failed = 0

    for image_path in image_paths:
        try:
            success = process_image(model, image_path)

            if success:
                succeeded += 1
            else:
                failed += 1

        except Exception as exc:
            failed += 1
            print(
                f"ERROR processing {image_path.name}: {exc}"
            )

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print(f"Successful: {succeeded}")
    print(f"Failed:     {failed}")
    print(f"Output:     {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
