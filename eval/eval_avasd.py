# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd Vamba
# export PYTHONPATH=.
from tools.vamba_chat import Vamba
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

RESIZE_WIDTH, RESIZE_HEIGHT = 360, 640 # TODO picked arbitrarily
FRAME_COUNT = 150 # TODO picked arbitrarily

df = pd.read_csv("/scratch/dvdai/AV-ASD/dataset/csvs/dataset.csv")
behaviors = df.columns[1:-1].tolist()  # Exclude Video_ID and Background

# Create ground truth lookup dictionary
ground_truth_dict = {}
for _, row in df.iterrows():
    video_id = row['Video_ID']
    gt_values = row[behaviors].values.astype(int).tolist()
    ground_truth_dict[video_id] = gt_values

folder_path = '/scratch/dvdai/AV-ASD/dataset/clips_video'
videos = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.mp4'):
            full_path = os.path.join(root, file)
            videos.append(full_path)

behavior_string = ',\n'.join(behaviors)

PROMPT = (
    "system\n"
    "You are a helpful assistant.\n"
    "user\n"
    f"You are analyzing an AV-ASD (Audio-Visual Autism Spectrum Dataset) video with {FRAME_COUNT} frames.\n\n"
    f"For each of the following 9 behaviors, indicate 1 if it is present and 0 if not present. The behaviors are:\n"
    f"{behavior_string}\n\n"
    "Output a list of 0 or 1, separated by a comma, one for each behavior. For example, if there are 9 behaviors, and only the first is observed, output 1,0,0,0,0,0,0,0,0\n\n"
    "Output only the list of 0s and 1s separated by commas. There should be 9 behaviors total."
)

model = Vamba(model_path="TIGER-Lab/Vamba-Qwen2-VL-7B", device="cuda")

print(f"TOTAL NUMBER OF VIDEOS: {len(videos)}")

results = []
all_predictions = []
all_ground_truths = []

def parse_output(output_str):
    """Parse the model output string to get binary predictions"""
    try:
        output_str = output_str.strip()
        print(output_str)
        predictions = [int(x.strip()) for x in output_str.split(',')]
        if len(predictions) != 9:
            if len(predictions) < 9:
                predictions.extend([0] * (9 - len(predictions)))
            else:
                predictions = predictions[:9]
        return predictions
    except:
        return [0] * 9

for idx, video_path in enumerate(videos):

    print(f"On prompt {idx}...")

    # Get video ID and ground truth
    video_id = Path(video_path).stem
    ground_truth = ground_truth_dict.get(video_id, [0]*9)

    test_input = [
        {
            "type": "video",
            "content": video_path,
            "metadata": {
                "video_num_frames": FRAME_COUNT,
                "video_sample_type": "middle",
                "img_longest_edge": RESIZE_HEIGHT,
                "img_shortest_edge": RESIZE_WIDTH,
            }
        },
        {
            "type": "text",
            "content": PROMPT
        }
    ]

    result = model(test_input)
    predictions = parse_output(result)

    # Store for metrics calculation
    all_predictions.append(predictions)
    all_ground_truths.append(ground_truth)

    # Calculate current metrics
    flat_preds = np.array(all_predictions).flatten()
    flat_truths = np.array(all_ground_truths).flatten()
    overall_accuracy = accuracy_score(flat_truths, flat_preds)
    overall_f1 = f1_score(flat_truths, flat_preds, average='binary', zero_division=0)

    # Per-category metrics
    category_metrics = []
    for i, behavior in enumerate(behaviors):
        cat_preds = [p[i] for p in all_predictions]
        cat_truths = [g[i] for g in all_ground_truths]
        cat_acc = accuracy_score(cat_truths, cat_preds)
        cat_f1 = f1_score(cat_truths, cat_preds, average='binary', zero_division=0)
        category_metrics.append({'behavior': behavior, 'accuracy': cat_acc, 'f1': cat_f1})

    # Store result with all info
    results.append({
        'Video_ID': video_id,
        'Output': result,
        'Predictions': ','.join(map(str, predictions)),
        'Ground_Truth': ','.join(map(str, ground_truth)),
        'Overall_Accuracy': overall_accuracy,
        'Overall_F1': overall_f1,
    })

    # Add per-category metrics to the result
    for cat_metric in category_metrics:
        results[-1][f"{cat_metric['behavior']}_acc"] = cat_metric['accuracy']
        results[-1][f"{cat_metric['behavior']}_f1"] = cat_metric['f1']

    # Save intermediate results
    results_df = pd.DataFrame(results)
    results_df.to_csv('vamba.csv', index=False)

    print(f"  Current Overall Accuracy: {overall_accuracy:.4f}, F1: {overall_f1:.4f}")

# Final save
results_df = pd.DataFrame(results)
results_df.to_csv('vamba.csv', index=False)

print(f"\nFinal Results:")
print(f"Overall Accuracy: {overall_accuracy:.4f}")
print(f"Overall F1 Score: {overall_f1:.4f}")
print("\nPer-Category Average Metrics:")
for cat_metric in category_metrics:
    print(f"  {cat_metric['behavior']}: Acc={cat_metric['accuracy']:.4f}, F1={cat_metric['f1']:.4f}")