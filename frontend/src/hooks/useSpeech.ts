"use client";

import { useState, useEffect, useRef } from "react";

export type VoiceSettings = {
  voiceName?: string;
  rate: number;  // speed: 1 = normal
  pitch: number; // tone: 1 = normal
  volume: number; // volume: 1 = max
  autoSpeakAnswers: boolean; // automatically speak AI responses
};

const DEFAULTS: VoiceSettings = {
  rate: 1,
  pitch: 1,
  volume: 1,
  autoSpeakAnswers: true,
};

export function useSpeech(initial?: Partial<VoiceSettings>) {
  // Browser checks
  const isBrowser = typeof window !== "undefined";

  const [supportedSTT, setSupportedSTT] = useState(false);
  const [supportedTTS, setSupportedTTS] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  const [settings, setSettings] = useState<VoiceSettings>({
    ...DEFAULTS,
    ...initial,
  });

  const recognitionRef = useRef<any>(null);

  // Load available voices (for TTS)
  useEffect(() => {
    if (!isBrowser) return;

    const synth = window.speechSynthesis;
    if (synth) {
      setSupportedTTS(true);
      const loadVoices = () => setVoices(synth.getVoices());
      loadVoices();
      synth.onvoiceschanged = loadVoices;
    }
  }, [isBrowser]);

  // Setup STT (SpeechRecognition)
  useEffect(() => {
    if (!isBrowser) return;

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      setSupportedSTT(true);
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";
      recognitionRef.current = recognition;
    }
  }, [isBrowser]);

  // Start listening to microphone
  const startListening = (onResult: (text: string) => void, onError?: (e: any) => void) => {
    const recognition = recognitionRef.current;
    if (!recognition) return;

    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results)
        .map((result: any) => result[0].transcript)
        .join("")
        .trim();
      onResult(transcript);
    };

    recognition.onerror = (e: any) => onError?.(e);
    recognition.onend = () => setIsListening(false);

    setIsListening(true);
    recognition.start();
  };

  const stopListening = () => {
    recognitionRef.current?.stop();
    setIsListening(false);
  };

  // Speak text aloud using TTS
  const speak = (text: string) => {
    if (!supportedTTS || !text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    if (settings.voiceName) {
      const voice = voices.find((v) => v.name === settings.voiceName);
      if (voice) utterance.voice = voice;
    }
    utterance.rate = settings.rate;
    utterance.pitch = settings.pitch;
    utterance.volume = settings.volume;

    window.speechSynthesis.cancel(); // Stop any previous speech
    window.speechSynthesis.speak(utterance);
  };

  return {
    supportedSTT,
    supportedTTS,
    isListening,
    voices,
    settings,
    setSettings,
    startListening,
    stopListening,
    speak,
  };
}

