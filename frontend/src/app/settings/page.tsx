// /src/app/settings/page.tsx
"use client";
import { useEffect } from "react";
import { useSpeech } from "@/hooks/useSpeech";

export default function SettingsPage() {
  const { voices, settings, setSettings, supportedTTS, supportedSTT } = useSpeech();

  useEffect(() => {
    // Persist to localStorage
    localStorage.setItem("voiceSettings", JSON.stringify(settings));
  }, [settings]);

  return (
    <section className="grid gap-6 max-w-2xl">
      <h1 className="text-2xl font-semibold">Settings</h1>
      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <p className="text-sm text-slate-300 mb-4">
          Speech Status:{" "}
          <span className={supportedSTT ? "text-emerald-400" : "text-rose-400"}>
            STT {supportedSTT ? "available" : "unavailable"}
          </span>{" "}
          •{" "}
          <span className={supportedTTS ? "text-emerald-400" : "text-rose-400"}>
            TTS {supportedTTS ? "available" : "unavailable"}
          </span>
        </p>

        <label className="block text-sm mb-2">Voice</label>
        <select
          className="w-full rounded border border-slate-700 bg-slate-950 p-2"
          value={settings.voiceName || ""}
          onChange={(e) => setSettings({ ...settings, voiceName: e.target.value || undefined })}
        >
          <option value="">Default</option>
          {voices.map((v) => (
            <option key={v.name} value={v.name}>
              {v.name} {v.lang ? `(${v.lang})` : ""}
            </option>
          ))}
        </select>

        <div className="grid grid-cols-3 gap-4 mt-4">
          <div>
            <label className="block text-sm mb-1">Rate</label>
            <input
              type="range" min={0.5} max={2} step={0.1}
              value={settings.rate}
              onChange={(e) => setSettings({ ...settings, rate: Number(e.target.value) })}
            />
            <div className="text-xs text-slate-400">{settings.rate.toFixed(1)}</div>
          </div>
          <div>
            <label className="block text-sm mb-1">Pitch</label>
            <input
              type="range" min={0} max={2} step={0.1}
              value={settings.pitch}
              onChange={(e) => setSettings({ ...settings, pitch: Number(e.target.value) })}
            />
            <div className="text-xs text-slate-400">{settings.pitch.toFixed(1)}</div>
          </div>
          <div>
            <label className="block text-sm mb-1">Volume</label>
            <input
              type="range" min={0} max={1} step={0.05}
              value={settings.volume}
              onChange={(e) => setSettings({ ...settings, volume: Number(e.target.value) })}
            />
            <div className="text-xs text-slate-400">{settings.volume.toFixed(2)}</div>
          </div>
        </div>

        <div className="mt-4 flex items-center gap-2">
          <input
            id="autoSpeak"
            type="checkbox"
            checked={settings.autoSpeakAnswers}
            onChange={(e) => setSettings({ ...settings, autoSpeakAnswers: e.target.checked })}
          />
          <label htmlFor="autoSpeak" className="text-sm">Auto-speak answers</label>
        </div>
      </div>
    </section>
  );
}
