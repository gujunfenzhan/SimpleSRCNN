import os
from pathlib import Path

filepath = Path("data/super_resolution/train")
for item in filepath.glob("*.jpg"):
    newFileName = filepath / (item.stem + ".png")
    if newFileName.exists():
        os.remove(newFileName)

    item.rename(newFileName)
