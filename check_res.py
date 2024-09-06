import json
from tqdm import tqdm


input_file_path = "/mnt/ve_share/LIV/datacomp/processed_data/caption_eval/881w_icc_score.jsonl"
output_file_path = "/mnt/ve_share/LIV/datacomp/processed_data/caption_eval/881w_icc_score_sort.jsonl"

batch_size = 1000000
retain_count = 100

# 读取所有数据
data = []
with open(input_file_path, "r", encoding="utf-8") as file:
    for line in tqdm(file, desc="Reading data"):
        data.append(json.loads(line))

# 按照score进行排序
data.sort(key=lambda x: x["score"], reverse=True)

# 准备输出文件
with open(output_file_path, "w", encoding="utf-8") as output_file:
    total_data_count = len(data)
    chunk_size = 1000000
    top_n = 1000000

    for start_idx in tqdm(range(0, total_data_count, chunk_size), desc="Processing chunks"):
        end_idx = min(start_idx + chunk_size, total_data_count)
        chunk = data[start_idx:end_idx]
        top_chunk = chunk[:top_n]

        for item in top_chunk:
            json_line = json.dumps(item, ensure_ascii=False)
            output_file.write(json_line + "\n")

print(f"Top {top_n} from every {chunk_size} records written to {output_file_path}")