from agent import Qtar
from midi_utils import convert_to_midi
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Generate music with Q-tar')
    parser.add_argument('--model_path', type=str, default='models/pretrained_qtar_model.pt',
                        help='Path to saved model')
    parser.add_argument('--output', type=str, default='qtar_solo.mid',
                        help='Output MIDI file path')
    parser.add_argument('--training_phase', type=int, default=2, help='Training phase')
    args = parser.parse_args()
    training_phase = args.training_phase
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    # Create Q-tar instance
    qtar = Qtar(
        scale='C_MAJOR',
        progression_type='I_VI_IV_V',
        training_phase=training_phase
    )
    chord_progression = qtar.chord_progression
    print("State size:", len(qtar.env._get_state()))
    print("Model input size:", qtar.model.shared_layers[0].in_features)
    print(f"Loading model from {args.model_path}")
    metadata = qtar.load_model(args.model_path)
    if metadata:
        print("Model metadata:", metadata)
        if 'total_motifs_learned' in metadata:
            print(f"Learned motifs: {metadata['total_motifs_learned']}")
    print("\nGenerating solo...")
    solo = qtar.generate_solo(training_phase=training_phase)
    # print solo
    print("\nGenerated Solo:")
    current_chord_idx = 0
    current_beat = 0
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    print("\nChord Progression:", ' | '.join(chord_progression))
    print("\nNote   Oct  Duration  Beat  Chord")
    print("-" * 45)
    for note, duration, beat in solo:
        if beat == 0 and current_beat != 0:
            current_chord_idx = (current_chord_idx + 1) % len(chord_progression)

        current_chord = chord_progression[current_chord_idx]
        octave = note // 12  # 0 or 1 for lower/upper octave
        note_in_octave = note % 12
        note_name = note_names[note_in_octave]
        print(f"{note_name:<6} {octave + 4:<4} {duration:<9.2f} {beat:<5.1f} {current_chord}")
        current_beat = beat
    # Save to MIDI file
    try:
        print(f"\nSaving to MIDI file: {args.output}")
        convert_to_midi(solo, chord_progression, args.output)
    except Exception as e:
        print(f"Could not save MIDI file: {str(e)}")
        print("Make sure pretty_midi is installed: pip install pretty_midi")
    # Print statistics
    total_notes = len(solo)
    unique_pitches = len(set(note for note, _, _ in solo))
    unique_pitch_classes = len(set(note % 12 for note, _, _ in solo))
    avg_duration = sum(duration for _, duration, _ in solo) / total_notes
    print(f"\nSolo statistics:")
    print(f"Total notes: {total_notes}")
    print(f"Unique pitches: {unique_pitches}")
    print(f"Unique pitch classes: {unique_pitch_classes}")
    print(f"Average note duration: {avg_duration:.2f} beats")
    # Analyze patterns if available
    if hasattr(qtar.env, 'motif_memory'):
        print(f"\nPattern Analysis:")
        print(f"Available motifs: {len(qtar.env.motif_memory)}")


if __name__ == "__main__":
    main()