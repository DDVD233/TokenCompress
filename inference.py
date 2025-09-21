# git clone https://github.com/TIGER-AI-Lab/Vamba
# cd Vamba
# export PYTHONPATH=.
from tools.vamba_chat import Vamba
model = Vamba(model_path="TIGER-Lab/Vamba-Qwen2-VL-7B", device="cuda")
test_input = [
    {
        "type": "video",
        "content": "assets/magic.mp4",
        "metadata": {
            "video_num_frames": 128,
            "video_sample_type": "middle",
            "img_longest_edge": 640,
            "img_shortest_edge": 256,
        }
    },
    {
        "type": "text",
        "content": "<video> Describe the magic trick."
    }
]
print(model(test_input))

test_input = [
    {
        "type": "image",
        "content": "assets/old_man.png",
        "metadata": {}
    },
    {
        "type": "text",
        "content": "<image> Describe this image."
    }
]
print(model(test_input))