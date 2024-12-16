import torch

# Chord definitions
CHORD_TONES = {
    'C': [0, 4, 7],    # C E G
    'Am': [9, 0, 4],   # A C E
    'F': [5, 9, 0],    # F A C
    'G': [7, 11, 2]    # G B D
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

RHYTHM_VALUES = [
            0.25,  # Sixteenth note
            0.5,  # Eighth note
            0.75,  # Dotted eighth note
            1.0,  # Quarter note
            1.5,  # Dotted quarter note
            2.0  # Half note
        ]