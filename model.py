import torch.nn as nn

class QtarNetwork(nn.Module):
    def __init__(self, state_size, note_size, rhythm_size):
        super(QtarNetwork, self).__init__()
        # Increase network capacity
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 512),  # Wider
            nn.LayerNorm(512),           # Add normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        # Separate note and rhythm paths with residual connections
        self.note_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, note_size)
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
        # Apply key mask
        if key_mask is not None:
            # Set logits of invalid notes to a very negative value
            masked_note_logits = note_logits.clone()
            masked_note_logits[:, :12] = note_logits[:, :12].masked_fill((key_mask == 0), float('-inf'))
        else:
            masked_note_logits = note_logits
        return masked_note_logits, rhythm_logits
