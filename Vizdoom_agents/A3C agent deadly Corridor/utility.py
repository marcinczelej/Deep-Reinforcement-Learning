from skimage import transform, io
import torch

left       = [1, 0, 0, 0, 0, 0, 0]
right      = [0, 1, 0, 0, 0, 0, 0]
shoot      = [0, 0, 1, 0, 0, 0, 0]
forward    = [0, 0, 0, 1, 0, 0, 0]
backward  =  [0, 0, 0, 0, 1, 0, 0]
turn_left =  [0, 0, 0, 0, 0, 1, 0]
turn_right    = [0, 0, 0, 0, 0, 0, 1]
actions = [left, right, shoot, forward, turn_left, turn_right]

def preprocess_frame(frame):
    cropped_frame = frame[30:-10,30:-30]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame, [100,120])
    preprocessed_frame = torch.from_numpy(preprocessed_frame).float().unsqueeze(0).unsqueeze(0)
    return preprocessed_frame

def healthReward(health_delta):
    if health_delta == 0:
        return 0
    if health_delta < 0:
        return -20

def killsReward(kills_delta):
    if kills_delta == 0:
        return 0
    if kills_delta > 0:
        return 100

