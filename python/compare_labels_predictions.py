from pathlib import Path

# Current project folder
ROOT_DIR = Path(
    r"C:\Users\R&D\source\repos\XRayAutoCentering\AutoCentering"
)

# Model 1: trained in the older project folder
MODEL_PATH_1 = Path(
    r"C:\Users\R&D\XRayAutoCentering\runs\obb\runs\obb"
    r"\xray_auto_centering_v1\weights\best.pt"
)

# Model 2: trained in the current project folder
MODEL_PATH_2 = (
    ROOT_DIR
    / "runs"
    / "obb"
    / "cable_476_v1"
    / "weights"
    / "best.pt"
)

# Images to test using both models
INPUT_DIR = ROOT_DIR / "input"

# Save comparison and fusion results
OUTPUT_DIR = ROOT_DIR / "output" / "rotated_box_fusion"

# Check that the required paths exist
print("Model 1 exists:", MODEL_PATH_1.exists(), MODEL_PATH_1)
print("Model 2 exists:", MODEL_PATH_2.exists(), MODEL_PATH_2)
print("Input folder exists:", INPUT_DIR.exists(), INPUT_DIR)