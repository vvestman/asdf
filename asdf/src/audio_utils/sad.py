#DO NOT USE THIS (Write/Use a better one)!

import numpy as np
from scipy.ndimage import maximum_filter1d


FRAME_SIZE = 0.02
MIN_DURATION = 300


def get_frame_size(fs):
    return int(FRAME_SIZE * fs)


def get_windowed_frames(signal, frame_size):
    num_frames = len(signal) // frame_size
    frames = np.stack([signal[i * frame_size: (i + 1) * frame_size] for i in range(num_frames)])
    return frames


def avg_filter(signal, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')


def endpoint_sad(signal, fs, E, threshold, filename):
    frame_size = get_frame_size(fs)
    sad = compute_sad_labels(E, threshold)

    # Filtering out small clutter near the ends
    if sad.size > 10:
        avg_sad = avg_filter(sad, 4) > 0.5
        indices_of_ones = np.where(avg_sad)[0]
        if indices_of_ones.size > 0:
            first = indices_of_ones[0]
            last = indices_of_ones[-1]
            first = max(0, first - 2)
            last = min(sad.size - 1, last + 3)
            sad[:first] = 0
            sad[last:] = 0

    indices_of_ones = np.where(sad)[0]
    if indices_of_ones.size > 0:
        first = indices_of_ones[0]
        # Removing start noise:
        if first == 0:
            index = 1
            while index < len(sad) and sad[index] == 1:
                index += 1
            length1 = index  # Number of speech frames in the beginning
            while index < len(sad) and sad[index] == 0:
                index += 1
            length2 = index - length1  # Number of non-speech frames after that
            if length2 > 2 * length1 and length1 < 0.3 * fs:  # If the number of non-speech frames is 2 times larger, the start is probably noise --> remove
                sad[:length1] = 0

    indices_of_ones = np.where(sad)[0]
    if indices_of_ones.size > 0:
        first = indices_of_ones[0]
        last = indices_of_ones[-1]
        sad = np.zeros_like(sad, dtype=bool)
        sad[first:last + 1] = 1
    else:
        sad = np.ones_like(sad, dtype=bool)
        print('ENDPOINT SAD WARNING: All sad labels are zeros! File: {}'.format(filename))
        print('--> Disabling SAD')

    one_count, sad = verify_labels(filename, sad)

    output_signal, sad_signal, E = create_cut_and_sad_signals(frame_size, one_count, sad, signal, E)

    return output_signal, sad_signal, E


def create_cut_and_sad_signals(frame_size, one_count, sad, signal, E):
    output_signal = np.zeros(one_count * frame_size)
    sad_signal = np.zeros_like(signal)
    speech_indices = np.where(sad)[0]
    for i, frameIndex in enumerate(speech_indices):
        output_signal[i * frame_size: (i + 1) * frame_size] = signal[frameIndex * frame_size: (frameIndex + 1) * frame_size]
        sad_signal[frameIndex * frame_size: (frameIndex + 1) * frame_size] = 1
    return output_signal, sad_signal, E[speech_indices]


def verify_labels(filename, sad):
    one_count = np.sum(sad)
    if one_count * FRAME_SIZE < 0.3:
        print('ENDPOINT SAD WARNING: Less than 300 ms of speech ({} frames only)! File: {}'.format(one_count, filename))
        print('--> Disabling SAD')
        sad = np.ones_like(sad, dtype=bool)
        one_count = sad.size
    return one_count, sad


def sad(signal, fs: int, E, threshold: float, max_break_duration: float, cut: bool, filename: str):
    frame_size = get_frame_size(fs)
    sad = compute_sad_labels(E, threshold)
    if (sad.size > 10):
        sad = avg_filter(sad, 5) > 0.5
    filter_size = int(max_break_duration / FRAME_SIZE)
    if filter_size > 1:
        sad = maximum_filter1d(sad, size=filter_size)
    one_count, sad = verify_labels(filename, sad)
    if cut:
        return create_cut_and_sad_signals(frame_size, one_count, sad, signal, E)
    else:
        output_signal = np.zeros_like(signal)
        sad_signal = np.zeros_like(signal)
        speech_indices = np.where(sad)[0]
        E[speech_indices] = 0
        for i, frameIndex in enumerate(speech_indices):
            output_signal[frameIndex * frame_size : (frameIndex + 1) * frame_size] = signal[frameIndex * frame_size : (frameIndex + 1) * frame_size]
            sad_signal[frameIndex * frame_size : (frameIndex + 1) * frame_size] = 1
        return output_signal, sad_signal, E


def compute_sad_energies(signal, fs):
    frame_size = get_frame_size(fs)
    frames = get_windowed_frames(signal, frame_size)
    E = 20 * np.log10(np.std(frames, axis=1) + np.finfo(float).eps)
    return E


def compute_sad_labels(E, threshold):
    return E > threshold


