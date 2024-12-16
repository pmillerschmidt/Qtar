
def convert_to_midi(melody, chord_progression, filename="qtar_solo.mid", base_note=60):
    # solo to midi
    try:
        import pretty_midi
    except ImportError:
        print("please install pretty_midi: pip install pretty_midi")
        return
    # guitar -> melody
    pm = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)
    current_time = 0.0
    for note, duration, _ in melody:
        # new note
        note_number = base_note + note
        note = pretty_midi.Note(
            velocity=100,
            pitch=note_number,
            start=current_time,
            end=current_time + duration
        )
        guitar.notes.append(note)
        current_time += duration
    # piano -> chords
    piano = pretty_midi.Instrument(program=0)  # 0 is acoustic grand piano
    chord_types = {
        '': [0, 4, 7],
        'm': [0, 3, 7],
        '7': [0, 4, 7, 10],
    }
    # Define root notes
    root_notes = {
        'C': 60,
        'D': 62,
        'E': 64,
        'F': 65,
        'G': 67,
        'A': 69,
        'B': 71,
    }
    # Add chords
    chord_time = 0.0
    beats_per_chord = 4.0  # Assuming 4 beats per chord
    for chord in chord_progression:
        # Parse chord name and type
        if chord.endswith('m'):
            root = chord[:-1]
            chord_type = 'm'
        elif chord.endswith('7'):
            root = chord[:-1]
            chord_type = '7'
        else:
            root = chord
            chord_type = ''
        # Get root note number
        root_note = root_notes[root]
        # add notes
        for note_offset in chord_types[chord_type]:
            note = pretty_midi.Note(
                velocity=80,
                pitch=root_note + note_offset,
                start=chord_time,
                end=chord_time + beats_per_chord
            )
            piano.notes.append(note)
        chord_time += beats_per_chord
    # add to midi file
    pm.instruments.append(guitar)
    pm.instruments.append(piano)
    pm.write(filename)
    print(f"MIDI file saved as {filename}")

