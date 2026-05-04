from collections import Counter
import json
with open("/home/olj3kor/praveen/chunk_methods/ragas_dataset/rag_gold_pipeline/output/stage_c_finalization/gold_v1.0.json") as f:
    data = json.load(f)

counts = Counter(len(r["reference_contexts"]) for r in data)
print(counts)