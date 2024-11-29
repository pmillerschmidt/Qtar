import { useRef } from 'react';

const NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const SECONDS_PER_BEAT = 0.5; // 120 BPM

export const AudioEngine = () => {
  const audioContext = useRef(null);
  const activeNodes = useRef([]);

  const CHORD_NOTES = {
    'C':  ['C', 'E', 'G'],
    'Am': ['A', 'C', 'E'],
    'F':  ['F', 'A', 'C'],
    'G':  ['G', 'B', 'D']
  };

  const init = () => {
    audioContext.current = new (window.AudioContext || window.webkitAudioContext)();
  };

  const stopAllNodes = () => {
    activeNodes.current.forEach(node => {
      try {
        node.stop();
        node.disconnect();
      } catch (e) {
        // Node might have already stopped
      }
    });
    activeNodes.current = [];
  };

  const playNote = (note, time, duration, isChord = false) => {
    const freq = 440 * Math.pow(2, (NOTES.indexOf(note) - 9) / 12);
    const osc = audioContext.current.createOscillator();
    const gain = audioContext.current.createGain();

    // Use different waveforms for melody and chords
    osc.type = isChord ? 'triangle' : 'sine';
    osc.frequency.value = freq;

    osc.connect(gain);
    gain.connect(audioContext.current.destination);

    const startTime = audioContext.current.currentTime + time;

    // Different volumes for melody and chords
    const maxGain = isChord ? 0.1 : 0.2;
    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(maxGain, startTime + 0.01);
    gain.gain.linearRampToValueAtTime(0, startTime + duration - 0.01);

    osc.start(startTime);
    osc.stop(startTime + duration);

    activeNodes.current.push(osc);
    return osc;
  };

  const playChord = (chord, time, duration) => {
    CHORD_NOTES[chord].forEach(note => {
      playNote(note, time, duration, true);
    });
  };

  const playPhrase = (phrase, startTime = 0) => {
    if (!audioContext.current) init();

    stopAllNodes();

    const currentTime = audioContext.current.currentTime;

    // Play melody notes
    phrase.notes
      .filter(note => (note.beat * SECONDS_PER_BEAT) >= startTime)
      .forEach(note => {
        playNote(
          note.note,
          (note.beat * SECONDS_PER_BEAT) - startTime,
          note.duration * SECONDS_PER_BEAT
        );
      });

    // Play chord progression
    phrase.chords.forEach((chord, i) => {
      playChord(
        chord,
        (i * 4 * SECONDS_PER_BEAT) - startTime,
        4 * SECONDS_PER_BEAT
      );
    });

    return currentTime;
  };

  return { playPhrase, stopAllNodes };
};

export default AudioEngine;