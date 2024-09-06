from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import jsonlines
import json
import os
from tqdm import tqdm

captions=[]
tokenizer = AutoTokenizer.from_pretrained("moranyanuka/icc")
model = AutoModelForSequenceClassification.from_pretrained("moranyanuka/icc").to("cuda")

all_captions=[]
all_scores=[]
cnt=0

# inputfile = "/mnt/share_disk/LIV/datacomp/processed_data/880w_1088w_dedup_processed/v0.1/demo-processed.jsonl"
inputfile = "/mnt/ve_share/songyuhao/dj_synth_challenge/output/image_captioning_output/mgm_pretrain_stage_1_online_res_2.jsonl"

total = sum(1 for line in open(inputfile, 'r', encoding='utf-8'))

with open(inputfile,"r",encoding='utf-8') as jfile:
    for i in tqdm(jfile):
        data=json.loads(i.strip())
        text = data["text"]
        text = text.replace("<__dj__image> ","")
        text = text.replace(" <|__dj__eoc|>","")
        cnt+=1

        captions.append(text)
        if len(captions)==500 or cnt==total:
            text_ids = tokenizer(captions, padding=True, return_tensors="pt", truncation=True).to("cuda")

            with torch.inference_mode():
                icc_scores = model(**text_ids)["logits"].view(-1).tolist()
            # print(icc_scores)
                all_captions.extend(captions)
                all_scores.extend(icc_scores)

                captions=[]

combined = zip(all_captions, all_scores)

# jsonl
with open("/mnt/ve_share/songyuhao/dj_synth_challenge/output/image_captioning_output/mgm_pretrain_stage_1_online_res_2_icc_score.jsonl", "w") as file:
    for name, score in combined:
        json_line = json.dumps({"caption": name, "score": score})
        file.write(json_line + "\n") 