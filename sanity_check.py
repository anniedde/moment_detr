import json

# Load predictions.json
with open("/playpen-nas-ssd4/awang/moment_detr/predictions.json", "r") as f:
    predictions = json.load(f)
pred_qids = set(entry["qid"] for entry in predictions)

# Load eval_transformed.jsonl
eval_qids = set()
with open("/playpen-nas-ssd4/awang/moment_detr/run_on_video/eval_transformed.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        eval_qids.add(entry["qid"])

# Find qids not present in both
only_in_predictions = pred_qids - eval_qids
only_in_eval = eval_qids - pred_qids

print("QIDs only in predictions.json:", sorted(only_in_predictions))
print("QIDs only in eval_transformed.jsonl:", sorted(only_in_eval))