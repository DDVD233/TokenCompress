# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd Vamba
# export PYTHONPATH=.
from tools.vamba_chat import Vamba
import json

RESIZE_WIDTH, RESIZE_HEIGHT = 360, 640

model = Vamba(model_path="TIGER-Lab/Vamba-Qwen2-VL-7B", device="cuda")

with open('data/output.json', 'r') as f:
    data = json.load(f)

results = []

for idx, item in enumerate(data):

    print(f"On prompt {idx}, module {item['module']}, test_type {item['test_type']}...")

    prompt = item['response'].split('Response')[0] # TODO fix hack

    print(prompt)

    test_input = [
        {
            "type": "video",
            "content": f"data/sample_1/{item['segment']}/{item['segment']}.mp4",
            "metadata": {
                "video_num_frames": item['num_frames'],
                "video_sample_type": "middle",
                "img_longest_edge": RESIZE_HEIGHT,
                "img_shortest_edge": RESIZE_WIDTH,
            }
        },
        {
            "type": "text",
            "content": prompt
        }
    ]

    result = item.copy()

    try:
        result['pred'] = model(test_input)
    except Exception as e:
        print(e)
        result['pred'] = 'N/A'

    print(f"Result: {result['pred']}\n")

    result['response'] = prompt + f'''
        Response:\nassistant\n{result['pred']}
    '''
    results.append(result)

with open('results/ados_output.json', 'w') as f:
    json.dump(results, f, indent = 4)
