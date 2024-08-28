import numpy as np

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len > max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def tile_to_size(x, size):
    x_len = x.shape[0]
    if x_len < size:
        num_repeats = int(size / x_len) + 1
        return np.tile(x, (1, num_repeats))[:, :size][0]
    return x

def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x