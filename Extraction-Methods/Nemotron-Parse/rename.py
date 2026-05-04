# from pathlib import Path

# folder = Path("/home/olj3kor/praveen/Nemo_NVIDIA/llm")

# for file in folder.iterdir():
#     if file.is_file() and file.stem.endswith("_llm"):
#         new_name = file.stem[:-4] + file.suffix
#         new_path = file.with_name(new_name)

#         # If target exists → remove it (overwrite behavior)
#         if new_path.exists():
#             new_path.unlink()

#         file.rename(new_path)

# print("Done. Files renamed and replaced.")


from pathlib import Path

folder = Path("/home/olj3kor/praveen/Structured_files/GLM_structured_Deterministic")

count = 0

for file in folder.iterdir():
    if file.is_file() and file.name.endswith("_hierarchy.json"):
        new_name = file.name.replace("_hierarchy.json", "_structure.json")
        new_path = file.with_name(new_name)
        file.rename(new_path)
        print(f"Renamed: {file.name} -> {new_name}")
        count += 1

print(f"\nDone! Renamed {count} file(s).")