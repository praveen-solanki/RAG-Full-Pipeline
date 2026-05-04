import os
import shutil

SRC_BASE = "/home/olj3kor/praveen/DocLing/MinerU_output/final_md"
DST_BASE = "/home/olj3kor/praveen/DocLing/MinerU_output/final_md_structured"

os.makedirs(DST_BASE, exist_ok=True)

for file in os.listdir(SRC_BASE):
    if file.endswith(".md"):
        name = file.replace(".md", "")
        
        src_file = os.path.join(SRC_BASE, file)
        dst_folder = os.path.join(DST_BASE, name)
        os.makedirs(dst_folder, exist_ok=True)
        
        dst_file = os.path.join(dst_folder, "doc.md")
        
        shutil.copy2(src_file, dst_file)

print("✅ New structured folder created safely")