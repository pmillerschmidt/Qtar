import torch.nn as nn


class QtarNetwork(nn.Module):
    def __init__(self, state_size, note_size, rhythm_size):
        super(QtarNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 512),  # state_size should be 15 for phase 2
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        # Note head now outputs 24 values instead of 12
        self.note_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, note_size)  # note_size should be 24
        )

        self.rhythm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, rhythm_size)
        )

    def forward(self, x, key_mask=None):
        shared_features = self.shared_layers(x)
        note_logits = self.note_head(shared_features)
        rhythm_logits = self.rhythm_head(shared_features)

        if key_mask is not None:
            masked_note_logits = note_logits.clone()
            for octave in range(2):
                start_idx = octave * 12
                end_idx = start_idx + 12
                masked_note_logits[:, start_idx:end_idx] = note_logits[:, start_idx:end_idx].masked_fill(
                    (key_mask == 0), float('-inf'))
            return masked_note_logits, rhythm_logits

        return note_logits, rhythm_logits