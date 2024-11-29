import torch

# Chord definitions
CHORD_TONES = {
    'C': [0, 4, 7],    # C E G
    'Am': [9, 0, 4],   # A C E
    'F': [5, 9, 0],    # F A C
    'G': [7, 11, 2]    # G B D
}

SCALE_DEGREES = {
    'C': [0, 2, 4, 5, 7, 9, 11],  # C major scale
    'Am': [9, 11, 0, 2, 4, 5, 7], # A minor scale
    'F': [5, 7, 9, 10, 0, 2, 4],  # F major scale
    'G': [7, 9, 11, 0, 2, 4, 6]   # G major scale
}

SCALE_MASKS = {
    'C_MAJOR': torch.tensor([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32),
    'G_MAJOR': torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1], dtype=torch.float32),
    'F_MAJOR': torch.tensor([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], dtype=torch.float32),
    'A_MINOR': torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32),
}

PROGRESSIONS = {
    'I_IV_V': ['C', 'F', 'G', 'C'],
    'II_V_I': ['Dm', 'G', 'C', 'C'],
    'I_VI_IV_V': ['C', 'Am', 'F', 'G'],
    'BLUES': ['C', 'C', 'C', 'C', 'F', 'F', 'C', 'C', 'G', 'F', 'C', 'G']
}