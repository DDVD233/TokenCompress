# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd Vamba
# export PYTHONPATH=.
from tools.vamba_chat import Vamba
import json
import os
import pandas as pd

RESIZE_WIDTH, RESIZE_HEIGHT = 360, 640 # TODO picked arbitrarily
FRAME_COUNT = 150 # TODO picked arbitrarily

df = pd.read_csv("/scratch/dvdai/AV-ASD/dataset/csvs/dataset.csv")
behaviors = df.columns[1:-1]

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

for idx, video in enumerate(videos):
    
    print(f"On prompt {idx}...")

    test_input = [
        {
            "type": "video",
            "content": video,
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
    results.append(result)

results_df = pd.DataFrame({
    'Video_ID' : [vid[:-4] for vid in videos],
    'Output' : results
})

results_df.to_csv('vamba.csv')
