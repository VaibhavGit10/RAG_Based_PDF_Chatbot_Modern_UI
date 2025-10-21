"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { askQuestion } from "@/lib/api";
import { useSpeech } from "@/hooks/useSpeech";

type Msg = { role: "user" | "assistant"; text: string; meta?: { sources?: any[] } };

const quickPrompts = [
  { label: "🔍 Key Clauses", prompt: "List the key clauses with brief summaries." },
  { label: "�� Dates", prompt: "Extract effective date, term, renewal and termination dates." },
  { label: "👥 Parties", prompt: "Who are the parties, roles, and any contact info?" },
  { label: "💵 Payment", prompt: "Summarize pricing, payment terms, and penalties." },
  { label: "🛑 Termination", prompt: "Summarize termination rights and notice periods." },
];

export default function ChatWithDocPage() {
  const router = useRouter();
  const { doc_id } = useParams<{ doc_id: string }>();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [talkMode, setTalkMode] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);

  // Voice (load saved settings if any)
  const initialVoice = useMemo(
    () => JSON.parse(globalThis?.localStorage?.getItem("voiceSettings") || "{}"),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );
  const speech = useSpeech(initialVoice);

  // Auto-scroll on new messages
  useEffect(() => {
    chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // If no doc_id, bounce to upload
  useEffect(() => {
    if (!doc_id) router.push("/upload");
  }, [doc_id, router]);

  const send = async (question: string) => {
    if (!question.trim() || !doc_id) return;
    const q = question.trim();
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    try {
      const res = await askQuestion(q, String(doc_id));
      if (res.status === "success") {
        const msg: Msg = { role: "assistant", text: res.answer || "", meta: { sources: res.sources || [] } };
        setMessages((m) => [...m, msg]);
        if (speech.settings.autoSpeakAnswers) speech.speak(msg.text);

        // Talk Mode: after speaking, start listening again
        if (talkMode && speech.supportedSTT) {
          // tiny pause to avoid capturing TTS
          setTimeout(() => {
            speech.startListening((heard) => {
              setInput(heard);
              setTimeout(() => send(heard), 150);
            });
          }, 600);
        }
      } else {
        setMessages((m) => [...m, { role: "assistant", text: res.message || "Something went wrong." }]);
      }
    } catch (e: any) {
      setMessages((m) => [...m, { role: "assistant", text: e?.message || "Failed to ask." }]);
    } finally {
      setLoading(false);
    }
  };

  const onMic = () => {
    if (!speech.supportedSTT) {
      alert("Speech Recognition is not supported in this browser.");
      return;
    }
    if (speech.isListening) {
      speech.stopListening();
    } else {
      speech.startListening((text) => {
        setInput(text);
        // auto-send after short delay
        setTimeout(() => send(text), 150);
      });
    }
  };

  const readLast = () => {
    const last = [...messages].reverse().find((m) => m.role === "assistant");
    if (last?.text) speech.speak(last.text);
  };

  return (
    <section className="mx-auto grid max-w-4xl gap-4">
      {/* Header row */}
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">
          Chat • <span className="text-sky-400">Document</span>{" "}
          <span className="text-slate-400 text-sm">({String(doc_id)})</span>
        </h1>
        <div className="ml-auto flex items-center gap-2">
          <button
            onClick={() => setTalkMode((v) => !v)}
            className={`rounded-md border px-3 py-1.5 text-sm ${
              talkMode
                ? "border-emerald-700 bg-emerald-900/40 text-emerald-300"
                : "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-600"
            }`}
            title="Talk Mode: auto listen after answers"
          >
            🗣️ {talkMode ? "Talk Mode: ON" : "Talk Mode: OFF"}
          </button>
          <button
            onClick={readLast}
            className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm hover:border-slate-600"
            title="Read last answer"
          >
            🔊 Read
          </button>
          <button
            onClick={() => router.push("/settings")}
            className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm hover:border-slate-600"
            title="Voice settings"
          >
            ⚙️ Voice
          </button>
        </div>
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap items-center gap-2">
        {quickPrompts.map((qp) => (
          <button
            key={qp.label}
            onClick={() => send(qp.prompt)}
            className="rounded-full border border-slate-800 bg-slate-900/60 px-3 py-1.5 text-xs text-slate-200 hover:border-slate-700"
          >
            {qp.label}
          </button>
        ))}
      </div>

      {/* Chat window */}
      <div
        ref={chatRef}
        className="h-[62vh] w-full overflow-y-auto rounded-xl border border-slate-800 bg-slate-900/40 p-4"
      >
        <div className="mx-auto grid max-w-3xl gap-3">
          <AnimatePresence initial={false}>
            {messages.map((m, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 8, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.98 }}
                transition={{ duration: 0.18 }}
                className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[78%] rounded-2xl px-4 py-2 shadow ${
                    m.role === "user"
                      ? "bg-sky-600 text-white"
                      : "bg-slate-800/90 border border-slate-700 text-slate-100"
                  }`}
                >
                  <p className="whitespace-pre-wrap leading-relaxed">{m.text}</p>

                  {/* Sources with confidence */}
                  {m.role === "assistant" && m.meta?.sources?.length ? (
                    <div className="mt-3 space-y-2">
                      <div className="text-xs text-slate-400">Sources:</div>
                      <div className="space-y-2">
                        {m.meta.sources.map((s: any, idx: number) => {
                          const text = typeof s === "string" ? s : s.text;
                          const score = typeof s?.score === "number" ? Math.round(s.score * 100) : null;
                          return (
                            <div
                              key={idx}
                              className="rounded-lg border border-slate-700 bg-slate-900/60 p-2 text-xs text-slate-300"
                            >
                              <div className="line-clamp-6">{text}</div>
                              <div className="mt-1 flex items-center gap-2">
                                {score !== null && (
                                  <span className="inline-flex items-center rounded bg-slate-800 px-2 py-0.5">
                                    Confidence: {score}%
                                  </span>
                                )}
                                {s?.filename && (
                                  <span className="text-slate-500">{s.filename}</span>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : null}
                </div>
              </motion.div>
            ))}

            {loading && (
              <motion.div
                key="typing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="max-w-[70%] rounded-2xl border border-slate-700 bg-slate-800/90 px-4 py-2 text-slate-300">
                  <span className="inline-flex items-center gap-2">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-slate-400 opacity-60" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-slate-400" />
                    </span>
                    Thinking…
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Input row */}
      <div className="mx-auto flex w-full max-w-3xl items-center gap-3">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => (e.key === "Enter" ? send(input) : undefined)}
          placeholder="Ask about this PDF (e.g., renewal terms, penalties, dates)…"
          className="flex-1 rounded-lg border border-slate-800 bg-slate-950 px-4 py-3 outline-none focus:border-sky-700"
        />
        <button
          onClick={() => send(input)}
          disabled={!input.trim() || loading}
          className="rounded-lg bg-sky-500 px-4 py-2 font-medium text-white hover:bg-sky-600 disabled:opacity-50"
        >
          Ask
        </button>
        <button
          onClick={onMic}
          className={`rounded-lg border px-4 py-2 ${
            speech.isListening
              ? "border-rose-600 bg-rose-900/40 text-rose-300"
              : "border-slate-700 bg-slate-900 text-slate-200 hover:border-slate-600"
          }`}
          title={speech.supportedSTT ? (speech.isListening ? "Stop" : "Speak") : "Voice not supported"}
        >
          🎤 {speech.isListening ? "Stop" : "Speak"}
        </button>
      </div>

      {/* Tips */}
      <div className="mx-auto w-full max-w-3xl text-center text-xs text-slate-400">
        Tip: Use <span className="text-sky-300">Talk Mode</span> for hands-free: ask → answer spoken → listens again.
      </div>
    </section>
  );
}

