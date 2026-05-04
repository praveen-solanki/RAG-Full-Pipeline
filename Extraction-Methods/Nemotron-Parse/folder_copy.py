import shutil
from pathlib import Path

src_root = Path("/home/olj3kor/praveen/DocLing/output")
dst_root = Path("/home/olj3kor/praveen/DocLing/structured")

# create destination folder if not exists
dst_root.mkdir(parents=True, exist_ok=True)

# glob pattern
files = src_root.glob("*/*.json")

for file in files:
    dst_file = dst_root / file.name
    shutil.copy(file, dst_file)

print("Done copying files.")