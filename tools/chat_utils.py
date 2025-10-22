import torchvision.io as tvio
import PIL
import torch
import numpy as np

from train.data import load_image_from_path, get_resize_output_image_size, get_frame_indices


def load_identity(data_path, patch_size, processor, **kwargs):
    if isinstance(data_path, tuple):
        return ([data_path[0]], data_path[1])
    else:
        return [data_path]

def load_media_data_image(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    image = load_image_from_path(data_path)
    if img_longest_edge is not None and img_shortest_edge is not None:
        resized_image = []
        for img in image:
            height, width = get_resize_output_image_size(img.size[1], img.size[0], img_shortest_edge, img_longest_edge)
            resized_image.append(img.resize((width, height), resample=3))
        image = resized_image
    return image

def load_media_data_video(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    max_img_seq_len = kwargs.get("max_img_seq_len", None)
    video_num_frames = kwargs.get("video_num_frames", 16)
    video_sample_type = kwargs.get("video_sample_type", "rand")
    do_resize = kwargs.get("do_resize", False)
    model_patch_size = patch_size

    # Read video metadata and frames using torchvision
    video_frames, audio, info = tvio.read_video(data_path, pts_unit='sec')
    vlen = video_frames.shape[0]
    fps = info['video_fps']

    if video_num_frames == 'auto':
        if not do_resize:
            height = video_frames.shape[1]
            width = video_frames.shape[2]
        else:
            height = processor.image_processor.size['height']
            width = processor.image_processor.size['width']
        num_patches = int((height // model_patch_size) * (width // model_patch_size))
        video_num_frames = int(max_img_seq_len // num_patches)

    frame_indices = get_frame_indices(video_num_frames, vlen, sample=video_sample_type, input_fps=fps)

    # Select frames based on indices
    selected_frames = video_frames[frame_indices]  # (T, H, W, C)
    # Convert from torch tensor (T, H, W, C) to numpy array
    frames_numpy = selected_frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    results = []
    for frame in frames_numpy:
        img = PIL.Image.fromarray(frame, mode="RGB")
        if img_shortest_edge is not None and img_longest_edge is not None:
            height, width = get_resize_output_image_size(img.size[1], img.size[0], img_shortest_edge, img_longest_edge)
            img = img.resize((width, height), resample=3)
        results.append(img)

    return [results]

def load_media_data_video_kangaroo(data_path, patch_size, processor, **kwargs):
    img_shortest_edge = kwargs.get("img_shortest_edge", None)
    img_longest_edge = kwargs.get("img_longest_edge", None)
    max_img_seq_len = kwargs.get("max_img_seq_len", None)
    video_num_frames = kwargs.get("video_num_frames", 16)
    video_sample_type = kwargs.get("video_sample_type", "rand")
    do_resize = kwargs.get("do_resize", False)
    model_patch_size = 14

    # Read video metadata and frames using torchvision
    video_frames, audio, info = tvio.read_video(data_path, pts_unit='sec')
    vlen = video_frames.shape[0]
    fps = info['video_fps']

    if video_num_frames == 'auto':
        if not do_resize:
            height = video_frames.shape[1]
            width = video_frames.shape[2]
        else:
            height = 448
            width = 448
        num_patches = int((height // model_patch_size) * (width // model_patch_size))
        video_num_frames = int(max_img_seq_len // num_patches)

    frame_indices = get_frame_indices(video_num_frames, vlen, sample=video_sample_type, input_fps=fps)
    durations = [idx / fps  for idx in frame_indices]

    # Select frames based on indices
    selected_frames = video_frames[frame_indices]  # (T, H, W, C)
    # Convert from torch tensor (T, H, W, C) to numpy array
    frames_numpy = selected_frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)
    results = []
    for frame in frames_numpy:
        img = PIL.Image.fromarray(frame, mode="RGB")
        if img_shortest_edge is not None and img_longest_edge is not None:
            height, width = get_resize_output_image_size(img, img_shortest_edge, img_longest_edge)
            img = img.resize((width, height), resample=3)
        results.append(img)

    return ([results], torch.Tensor(durations))