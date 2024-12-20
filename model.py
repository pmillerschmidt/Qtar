import torch
import torch.nn as nn


class QtarNetwork(nn.Module):
    def __init__(self, state_size, note_size, rhythm_size):
        super(QtarNetwork, self).__init__()
        # shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        # pitch (0-12 pitches)
        self.pitch_class_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        # octave (1-2 octaves)
        self.octave_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # rhythm head
        self.rhythm_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, rhythm_size)
        )

    def forward(self, x):
        features = self.shared_layers(x)
        # predict pitch and octave
        pitch_logits = self.pitch_class_head(features)
        octave_logits = self.octave_head(features)
        # combine into note prediction
        note_logits = self._combine_pitch_octave(pitch_logits, octave_logits)
        # rhythm prediction
        rhythm_logits = self.rhythm_head(features)
        return note_logits, rhythm_logits

    def _combine_pitch_octave(self, pitch_logits, octave_logits):
        batch_size = pitch_logits.size(0)
        full_logits = torch.zeros(batch_size, 24).to(pitch_logits.device)
        # pitch probabilities across octaves
        for octave in range(2):
            start_idx = octave * 12
            end_idx = start_idx + 12
            octave_weight = torch.sigmoid(octave_logits[:, octave]).unsqueeze(-1)
            full_logits[:, start_idx:end_idx] = pitch_logits * octave_weight
        return full_logits