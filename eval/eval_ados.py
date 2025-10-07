# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd Vamba
# export PYTHONPATH=.
from tools.vamba_chat import Vamba
import json

RESIZE_WIDTH, RESIZE_HEIGHT = 360, 640 # TODO picked arbitrarily
FRAME_COUNT = 150 # TODO picked arbitrarily

SEGMENTS = [
    'anticipation_of_routine_1',
    'anticipation_of_routine_2',
    'anticipation_of_routine_3',
    'anticipation_of_routine_4',
    'anticipation_of_routine_5',
    'anticipation_of_routine_6',
    'anticipation_of_routine_7',
    'bubble_play',
    'free_play',
    'response_to_joint_attention_1',
    'response_to_joint_attention_2',
    'response_to_name',
    'responsive_social_smile_1',
    'responsive_social_smile_2',
    'responsive_social_smile_3',
    'responsive_social_smile_4',
]

def get_prompts(json_obj):

    labels_str = "\n".join(json_obj["labels"])

    response = (
        "system\n"
        "You are a helpful assistant.\n"
        "user\n"
        f"You are analyzing an ADOS (Autism Diagnostic Observation Schedule) video with {FRAME_COUNT} frames.\n\n"
        f"Module: {json_obj['module']}\n"
        f"Test: {json_obj['test type']}\n"
        f"Task: {json_obj['description']}\n\n"
        f"Labels:\n{labels_str}\n\n"
        "Provide your answer in a single number and nothing else."
    )

    return [
        {
            'module' : json_obj['module'],
            'test_type' : json_obj['test type'],
            'response' : response,
            'num_frames' : FRAME_COUNT,
            'use_aks' : True,
            'segment' : segment
        }

        for segment in SEGMENTS
    ]

model = Vamba(model_path="TIGER-Lab/Vamba-Qwen2-VL-7B", device="cuda")

with open('data/autism_scoring.json', 'r') as f:
    data = json.load(f)

results = []
prompts = []

for item in data:
    prompts.extend(get_prompts(item))

print(f"TOTAL NUMBER OF PROMPTS: {len(prompts)}")

for idx, item in enumerate(prompts):
    
    print(f"On prompt {idx}, module {item['module']}, test_type {item['test_type']}, segment {item['segment']}...")
    prompt = item['response']
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
