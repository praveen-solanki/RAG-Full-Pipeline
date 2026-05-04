import os

BASE_DIR = "./results"
OUTPUT_FILE = "combined_all_results.txt"

def combine_all_results():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for root, dirs, files in os.walk(BASE_DIR):
            for file in sorted(files):
                if file.endswith(".txt") and file != "summary.txt":
                    
                    file_path = os.path.join(root, file)

                    # Write separator + file info
                    outfile.write("\n" + "="*100 + "\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write("="*100 + "\n\n")

                    # Append content
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")

    print(f"\n✅ Combined file created: {OUTPUT_FILE}")


if __name__ == "__main__":
    combine_all_results()