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

    def forward(self, x):
        shared_features = self.shared_layers(x)
        note_output = self.note_head(shared_features)
        rhythm_output = self.rhythm_head(shared_features)
        return note_output, rhythm_output
