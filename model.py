import torch.nn as nn

class QtarNetwork(nn.Module):
    def __init__(self, state_size, note_size, rhythm_size):
        super(QtarNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.note_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, note_size)
        )
        self.rhythm_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, rhythm_size)
        )

    def forward(self, x, key_mask=None):
        shared_features = self.shared_layers(x)
        note_logits = self.note_head(shared_features)
        rhythm_logits = self.rhythm_head(shared_features)

        # Apply key mask
        if key_mask is not None:
            # Set logits of invalid notes to a very low value
            masked_note_logits = note_logits + (key_mask.log() - key_mask.log())
        else:
            masked_note_logits = note_logits

        return masked_note_logits, rhythm_logits
