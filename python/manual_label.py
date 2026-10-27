
import cv2
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT_DIR / "dataset" / "raw_images"
LABEL_DIR = ROOT_DIR / "dataset" / "labels" / "train"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp",
    ".tif", ".tiff", ".webp"
}

CLASS_ID = 0
WINDOW_NAME = "Manual YOLO OBB Labeling"

MAX_DISPLAY_WIDTH = 1200
MAX_DISPLAY_HEIGHT = 800

# Colors use OpenCV BGR format
BG_COLOR = (30, 30, 30)
CYAN = (255, 220, 50)
GREEN = (80, 230, 130)
WHITE = (245, 245, 245)
YELLOW = (0, 230, 255)
RED = (70, 70, 255)

# ============================================================
# PREPARE FOLDERS AND IMAGE LIST
# ============================================================

LABEL_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image folder not found:\n{IMAGE_DIR}"
    )

image_files = sorted(
    [
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ],
    key=lambda p: p.name.lower()
)

if not image_files:
    raise RuntimeError(
        f"No images found in:\n{IMAGE_DIR}"
    )

# ============================================================
# STATE
# ============================================================

current_index = 0
points = []
current_image = None
display_image = None

scale = 1.0
image_width = 0
image_height = 0

message = "Ready"
message_color = GREEN
window_ready = False


# ============================================================
# LABEL HELPERS
# ============================================================

def label_path_for(image_path):
    return LABEL_DIR / f"{image_path.stem}.txt"


def load_existing_label(image_path):
    path = label_path_for(image_path)

    if not path.exists():
        return []

    try:
        lines = [
            line.strip()
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not lines:
            return []

        parts = lines[0].split()

        if len(parts) != 9:
            print(f"Unsupported label format: {path.name}")
            return []

        coords = [float(v) for v in parts[1:]]

        if not all(0.0 <= v <= 1.0 for v in coords):
            print(f"Coordinates outside 0-1: {path.name}")
            return []

        loaded_points = []

        for i in range(0, 8, 2):
            x = round(coords[i] * (image_width - 1))
            y = round(coords[i + 1] * (image_height - 1))
            loaded_points.append((x, y))

        return loaded_points

    except (ValueError, OSError) as exc:
        print(f"Could not load {path.name}: {exc}")
        return []


def save_label():
    global message, message_color

    if len(points) != 4:
        message = "ERROR: Select all 4 corners before saving"
        message_color = RED
        return False

    values = []

    for x, y in points:
        nx = x / max(1, image_width - 1)
        ny = y / max(1, image_height - 1)

        values.extend([
            min(1.0, max(0.0, nx)),
            min(1.0, max(0.0, ny))
        ])

    output = str(CLASS_ID) + " " + " ".join(
        f"{v:.8f}" for v in values
    )

    path = label_path_for(image_files[current_index])

    try:
        path.write_text(output + "\n", encoding="utf-8")
        message = f"SAVED: {path.name}"
        message_color = GREEN
        print(f"Saved: {path}")
        return True

    except OSError as exc:
        message = f"SAVE FAILED: {exc}"
        message_color = RED
        print(message)
        return False


def delete_label():
    global message, message_color

    path = label_path_for(image_files[current_index])

    try:
        if path.exists():
            path.unlink()
            message = f"Deleted: {path.name}"
        else:
            message = "No saved label to delete"

        message_color = YELLOW

    except OSError as exc:
        message = f"DELETE FAILED: {exc}"
        message_color = RED


# ============================================================
# IMAGE LOADING
# ============================================================

def load_current_image():
    global current_image, display_image, scale
    global image_width, image_height, points
    global message, message_color

    path = image_files[current_index]

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Cannot open image: {path}")

    current_image = image
    image_height, image_width = image.shape[:2]

    scale = min(
        MAX_DISPLAY_WIDTH / image_width,
        MAX_DISPLAY_HEIGHT / image_height,
        1.0
    )

    display_width = max(1, round(image_width * scale))
    display_height = max(1, round(image_height * scale))

    display_image = cv2.resize(
        current_image,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA
    )

    points = load_existing_label(path)

    if len(points) == 4:
        message = "Existing label loaded. Press S to save or N for next."
    else:
        message = "Click 4 object corners in order."

    message_color = CYAN


# ============================================================
# DRAW SCREEN
# ============================================================

def draw_screen():
    canvas = display_image.copy()
    height, width = canvas.shape[:2]

    # Draw polygon edges
    if len(points) >= 2:
        for i in range(len(points) - 1):
            p1 = (
                round(points[i][0] * scale),
                round(points[i][1] * scale)
            )
            p2 = (
                round(points[i + 1][0] * scale),
                round(points[i + 1][1] * scale)
            )

            cv2.line(canvas, p1, p2, GREEN, 2, cv2.LINE_AA)

    if len(points) == 4:
        p1 = (
            round(points[3][0] * scale),
            round(points[3][1] * scale)
        )
        p2 = (
            round(points[0][0] * scale),
            round(points[0][1] * scale)
        )

        cv2.line(canvas, p1, p2, GREEN, 2, cv2.LINE_AA)

    # Draw corner markers and numbers
    for i, (x, y) in enumerate(points):
        dx = round(x * scale)
        dy = round(y * scale)

        cv2.circle(canvas, (dx, dy), 7, GREEN, -1, cv2.LINE_AA)
        cv2.circle(canvas, (dx, dy), 10, WHITE, 1, cv2.LINE_AA)

        cv2.putText(
            canvas,
            str(i + 1),
            (dx + 10, dy - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            YELLOW,
            2,
            cv2.LINE_AA
        )

    # Top information bar
    cv2.rectangle(canvas, (0, 0), (width, 38), BG_COLOR, -1)

    label_file = label_path_for(image_files[current_index])
    status = "LABEL EXISTS" if label_file.exists() else "NOT LABELLED"

    header = (
        f"Image {current_index + 1}/{len(image_files)}"
        f" | {status} | Points: {len(points)}/4"
    )

    cv2.putText(
        canvas, header, (12, 25),
        cv2.FONT_HERSHEY_DUPLEX, 0.58,
        WHITE, 1, cv2.LINE_AA
    )

    # Bottom status bar
    cv2.rectangle(
        canvas, (0, height - 58), (width, height), BG_COLOR, -1
    )

    cv2.putText(
        canvas,
        message[:110],
        (12, height - 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        message_color,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "S Save | N Next | P Previous | U Undo | R Reset | D Delete | Q Quit",
        (12, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.imshow(WINDOW_NAME, canvas)


# ============================================================
# MOUSE INPUT
# ============================================================

def mouse_callback(event, x, y, flags, userdata):
    global points, message, message_color

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if len(points) >= 4:
        message = "Four corners selected. Press S to save or R to reset."
        message_color = YELLOW
        return

    original_x = round(x / scale)
    original_y = round(y / scale)

    original_x = max(0, min(image_width - 1, original_x))
    original_y = max(0, min(image_height - 1, original_y))

    points.append((original_x, original_y))

    message = f"Corner {len(points)}/4 selected"
    message_color = CYAN


# ============================================================
# NAVIGATION
# ============================================================

def move_to(index):
    global current_index, message, message_color

    new_index = index % len(image_files)

    try:
        current_index = new_index
        load_current_image()

    except Exception as exc:
        message = f"IMAGE ERROR: {exc}"
        message_color = RED
        print(message)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global points, message, message_color, window_ready

    print("\n========== MANUAL YOLO OBB LABELING ==========")
    print("Images :", IMAGE_DIR)
    print("Labels :", LABEL_DIR)
    print("Total  :", len(image_files))
    print("Click four corners in perimeter order.")
    print("S=Save | N=Next | P=Previous | U=Undo")
    print("R=Reset | D=Delete label | Q/Esc=Quit")
    print("==============================================\n")

    try:
        load_current_image()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            WINDOW_NAME,
            min(MAX_DISPLAY_WIDTH, round(image_width * scale)),
            min(MAX_DISPLAY_HEIGHT, round(image_height * scale))
        )
        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
        window_ready = True

        while True:
            draw_screen()

            # waitKeyEx captures keyboard input from the focused OpenCV window.
            key = cv2.waitKeyEx(30)

            if key == -1:
                continue

            # Normalize key codes, including uppercase and extended key codes.
            key = key & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            elif key in (ord("s"), ord("S")):
                save_label()

            elif key in (ord("n"), ord("N")):
                # Do not silently discard a partially labelled image.
                if 0 < len(points) < 4:
                    message = "Finish 4 corners, or press R to reset before next."
                    message_color = RED
                    continue

                if len(points) == 4 and not save_label():
                    continue

                if current_index == len(image_files) - 1:
                    message = "Last image reached. Press Q to quit or N to restart."
                    message_color = YELLOW
                    continue

                move_to(current_index + 1)

            elif key in (ord("p"), ord("P")):
                if current_index > 0:
                    move_to(current_index - 1)
                else:
                    message = "Already at the first image"
                    message_color = YELLOW

            elif key in (ord("u"), ord("U")):
                if points:
                    points.pop()
                    message = "Last corner undone"
                    message_color = CYAN
                else:
                    message = "No corner to undo"
                    message_color = YELLOW

            elif key in (ord("r"), ord("R")):
                points = []
                message = "Reset. Click the four corners again."
                message_color = CYAN

            elif key in (ord("d"), ord("D")):
                delete_label()
                points = []

    except Exception as exc:
        print(f"\nERROR: {exc}")

    finally:
        if window_ready:
            cv2.destroyAllWindows()

    print("\nLabeling session ended.")
    print("Labels saved in:", LABEL_DIR)


if __name__ == "__main__":
    main()
