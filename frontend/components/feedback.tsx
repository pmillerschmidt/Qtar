"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Star, AlertCircle, Play, Pause, Volume2, VolumeX, ZoomIn, ZoomOut } from 'lucide-react';
import { AudioEngine } from './AudioEngine';
import PianoRoll from './PianoRoll';

// Musical constants
const NOTES = [
  // Octave 4
  'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4',
  // Octave 5
  'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5'
];

const MIDI_START = 60; // Middle C (C4)
const TOTAL_NOTES = 24; // Two octaves
const SECONDS_PER_BEAT = 0.5; // 120 BPM
const PIXELS_PER_BEAT = 40;
const NOTE_HEIGHT = 20;

// Chord definitions with octaves
const CHORD_NOTES = {
  'C':  ['C4', 'E4', 'G4'],
  'Am': ['A4', 'C4', 'E4'],
  'F':  ['F4', 'A4', 'C5'],
  'G':  ['G4', 'B4', 'D5']
};

const FeedbackInterface = () => {
  const [currentPhrase, setCurrentPhrase] = useState(0);
  const [ratings, setRatings] = useState({});
  const [showHelp, setShowHelp] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentBeat, setCurrentBeat] = useState(null);
  const [isLoadingMelody, setIsLoadingMelody] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  // Add this new state
  const [isTraining, setIsTraining] = useState(false);
  const [phrases, setPhrases] = useState([]);
  const [submitStatus, setSubmitStatus] = useState<string>('');

  const audioEngine = useRef(AudioEngine()).current;
  const playbackTimer = useRef(null);
  const audioStartTime = useRef(null);
  const elapsedTime = useRef(0);

  const fetchPhrase = async () => {
    try {
      const response = await fetch('/api/feedback');
      if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
      const data = await response.json();
        if (data.status === 'ready') {
            setPhrases([data]);
        }
    } catch (error) {
      console.error('Failed to fetch phrase:', error);
    }
  };

  // Add this useEffect to fetch initial phrase when component mounts
  useEffect(() => {
    fetchPhrase();
  }, []);

  const handleRating = (phraseIndex, rating) => {
    setRatings(prev => ({
      ...prev,
      [phraseIndex]: rating
    }));
  };

  const startTraining = async () => {
    setIsTraining(true);
    setIsLoadingMelody(true);
    try {
        // Start training
        const response = await fetch('/api/feedback', {
            method: 'PUT'
        });
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Keep polling until we get a new melody
        let newMelody = null;
        while (!newMelody) {
            const melodyResponse = await fetch('/api/feedback');
            if (!melodyResponse.ok) {
                throw new Error(`HTTP error! status: ${melodyResponse.status}`);
            }
            const melodyData = await melodyResponse.json();

            // Check if we have a valid melody
            if (melodyData.status === 'ready' && melodyData.notes && melodyData.notes.length > 0) {
                newMelody = melodyData;
                setPhrases([newMelody]);
                break;
            }

            // Wait a bit before polling again
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
    } catch (error) {
        console.error('Training failed:', error);
    } finally {
        setIsTraining(false);
        setIsLoadingMelody(false);
    }
};

  const playPhrase = (phraseIndex) => {
    if (isPlaying) {
      // Stop playback
      audioEngine.stopAllNodes();
      clearInterval(playbackTimer.current);
      setIsPlaying(false);
      setCurrentBeat(null);
      elapsedTime.current = 0;
      return;
    }

    // Start new playback
    setIsPlaying(true);
    audioStartTime.current = Date.now();
    let beat = 0;

    // Start playback
    audioEngine.playPhrase(phrases[phraseIndex]);

    // Update beat counter for visualization
    playbackTimer.current = setInterval(() => {
      if (beat >= 16) {
        setIsPlaying(false);
        setCurrentBeat(null);
        clearInterval(playbackTimer.current);
        elapsedTime.current = 0;
      } else {
        setCurrentBeat(beat);
        beat += 0.25;
      }
    }, SECONDS_PER_BEAT * 250); // Update 4 times per beat
  };

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (playbackTimer.current) {
        clearInterval(playbackTimer.current);
      }
      audioEngine.stopAllNodes();
    };
  }, []);

  const fetchCurrentSolo = async () => {
  try {
    const response = await fetch('/api/feedback');  // Use the Next.js API route
    if (response.ok) {
      const data = await response.json();
      setPhrases([data]); // Update with current solo
    }
  } catch (error) {
    console.error('Failed to fetch current solo:', error);
  }
};

  const submitFeedback = async () => {
      setIsLoadingMelody(true);
      try {
          // First submit feedback
          const feedbackResponse = await fetch('/api/feedback', {
              method: 'POST',
              headers: {
                  'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                  rating: ratings[0]
              }),
          });

          if (feedbackResponse.ok) {
              // Then trigger training
              const response = await fetch('/api/feedback', {
                  method: 'PUT'
              });
              if (!response.ok) {
                  throw new Error(`HTTP error! status: ${response.status}`);
              }

              // Keep polling until we get a new melody
              let newMelody = null;
              while (!newMelody) {
                  const melodyResponse = await fetch('/api/feedback');
                  if (!melodyResponse.ok) {
                      throw new Error(`HTTP error! status: ${melodyResponse.status}`);
                  }
                  const melodyData = await melodyResponse.json();

                  // Check if we have a valid melody
                  if (melodyData.status === 'ready' && melodyData.notes && melodyData.notes.length > 0) {
                      newMelody = melodyData;
                      setPhrases([newMelody]);
                      setRatings({}); // Clear ratings for new melody
                      break;
                  }

                  // Wait a bit before polling again
                  await new Promise(resolve => setTimeout(resolve, 1000));
              }

              setSubmitStatus('Feedback submitted successfully!');
              setTimeout(() => {
                  setSubmitStatus('');
              }, 3000);
          }
      } catch (error) {
          console.error('Failed to submit feedback or get new melody:', error);
          setSubmitStatus('Failed to submit feedback');
      } finally {
          setIsLoadingMelody(false);
      }
  };

  // Poll for new solos periodically
  useEffect(() => {
    const interval = setInterval(fetchCurrentSolo, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-100">
      {/* Fixed Header */}
      <header className="flex-none bg-white border-b">
        <div className="max-w-5xl mx-auto w-full px-6 py-3">
          <div className="flex justify-between items-center">
            <h1 className="text-xl font-bold">Q-tar Feedback Interface</h1>
            <button
              onClick={() => setShowHelp(!showHelp)}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <AlertCircle className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Scrollable Main Content */}
      <main className="flex-1 overflow-auto px-6 py-4">
        <div className="max-w-3xl mx-auto w-full">
          {/* Help Section */}
          {showHelp && (
              <div className="bg-blue-50 p-4 rounded-lg text-sm mb-4">
                <h2 className="font-semibold mb-2">Rating Guide:</h2>
                <ul className="space-y-1 list-disc pl-5">
                  <li>1 Star: Poor - Doesn't sound musical</li>
                  <li>2 Stars: Fair - Basic musical structure but needs improvement</li>
                  <li>3 Stars: Good - Decent musical phrase</li>
                  <li>4 Stars: Very Good - Engaging and well-structured</li>
                  <li>5 Stars: Excellent - Outstanding musical quality</li>
                </ul>
              </div>
          )}

          {/* Phrases */}
          <div className="space-y-4">
            {phrases.map((phrase, phraseIndex) => (
                <div
                    key={phraseIndex}
                    className="bg-white rounded-lg shadow-sm border p-4"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold">
                      Phrase {phraseIndex + 1}
                    </h2>
                    <button
                        onClick={() => setIsMuted(!isMuted)}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                    >
                      {isMuted ? <VolumeX className="w-5 h-5"/> : <Volume2 className="w-5 h-5"/>}
                    </button>
                  </div>

                  {/* Piano Roll Container with Loading State */}
                  <div className="w-full overflow-x-auto bg-gray-50 rounded-lg p-2">
                    {isLoadingMelody ? (
                        <div className="min-h-[300px] flex items-center justify-center">
                          <div
                              className="animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
                        </div>
                    ) : (
                        <div className="min-w-[500px]">
                          <PianoRoll
                              phrase={phrase}
                              isPlaying={isPlaying}
                              currentBeat={currentBeat}
                          />
                        </div>
                    )}
                  </div>
                </div>
            ))}
          </div>
        </div>
      </main>

      {/* Fixed Footer */}
      <footer className="flex-none bg-white border-t pb-6">
        <div className="max-w-5xl mx-auto w-full px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            {/* Train Model */}
            <button
                onClick={startTraining}
                disabled={isTraining}
                className="bg-purple-500 hover:bg-purple-600 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
            >
              {isTraining ? 'Training...' : 'Train Model'}
            </button>

            {/* Play Button */}
            <button
                onClick={() => playPhrase(0)}
                className="flex items-center gap-2 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {isPlaying ? (
                  <>
                    <Pause className="w-5 h-5"/>
                    <span>Pause</span>
                  </>
              ) : (
                  <>
                    <Play className="w-5 h-5"/>
                    <span>Play</span>
                  </>
              )}
            </button>

            {/* Star Rating */}
            <div className="flex items-center gap-1">
              {[1, 2, 3, 4, 5].map((rating) => (
                  <button
                      key={rating}
                      onClick={() => handleRating(0, rating)}
                      className={`p-1.5 rounded-full transition-colors ${
                          ratings[0] === rating
                              ? 'text-yellow-500'
                              : 'text-gray-300 hover:text-yellow-500'
                      }`}
                  >
                    <Star
                        className="w-6 h-6"
                        fill={ratings[0] >= rating ? 'currentColor' : 'none'}
                    />
                  </button>
              ))}
            </div>

            {/* Submit Feedback */}
            <div className="flex items-center gap-4">
              {submitStatus && (
                  <span
                      className={`text-sm ${
                          submitStatus.includes('Failed') ? 'text-red-500' : 'text-green-500'
                      }`}
                  >
                  {submitStatus}
                </span>
              )}
              <button
                  onClick={submitFeedback}
                  disabled={Object.keys(ratings).length !== phrases.length}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors disabled:opacity-50"
              >
                Submit Feedback
              </button>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default FeedbackInterface;