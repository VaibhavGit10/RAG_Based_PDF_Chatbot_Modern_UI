// /src/app/page.tsx
"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { health } from "@/lib/api";

export default function HomePage() {
  const [api, setApi] = useState<"ok" | "down" | "checking">("checking");

  useEffect(() => {
    let mounted = true;
    health()
      .then((r) => mounted && setApi(r?.status === "ok" ? "ok" : "down"))
      .catch(() => mounted && setApi("down"));
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <section className="mx-auto grid max-w-4xl gap-8">
      <div className="rounded-2xl border border-slate-800 bg-gradient-to-b from-slate-900 to-slate-950 p-8 shadow-xl">
        <h1 className="text-3xl font-bold tracking-tight">
          Chat with your PDFs using <span className="text-sky-400">RAG + Qdrant + Groq</span>
        </h1>
        <p className="mt-3 text-slate-300">
          Upload a PDF, we’ll chunk & embed it locally (ONNX, CPU), store vectors in Qdrant, and
          answer your questions using Groq — fast, cost-efficient, and private.
        </p>

        <div className="mt-6 flex items-center gap-3">
          <Link
            href="/upload"
            className="rounded-lg bg-sky-500 px-5 py-2.5 font-medium text-white hover:bg-sky-600"
          >
            Get started
          </Link>
          <Link
            href="/settings"
            className="rounded-lg border border-slate-700 px-5 py-2.5 text-slate-200 hover:border-slate-600"
          >
            Settings
          </Link>
          <span
            className={`ml-auto inline-flex items-center gap-2 rounded-md border px-3 py-1.5 text-xs ${
              api === "ok"
                ? "border-emerald-700 bg-emerald-900/30 text-emerald-300"
                : api === "down"
                ? "border-rose-700 bg-rose-900/30 text-rose-300"
                : "border-slate-700 bg-slate-800 text-slate-300"
            }`}
            title="Backend health"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                api === "ok" ? "bg-emerald-400" : api === "down" ? "bg-rose-400" : "bg-slate-400"
              }`}
            />
            {api === "ok" ? "Backend: OK" : api === "down" ? "Backend: Down" : "Checking…"}
          </span>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {[
          {
            title: "Local Embeddings",
            desc: "ONNX MiniLM (CPU) – no GPU, no PyTorch.",
          },
          {
            title: "Qdrant Vectors",
            desc: "Fast, filterable search over your chunks.",
          },
          {
            title: "Groq Answers",
            desc: "LLM answers grounded in your PDF context.",
          },
        ].map((c) => (
          <div
            key={c.title}
            className="rounded-xl border border-slate-800 bg-slate-900/40 p-4"
          >
            <h3 className="font-medium">{c.title}</h3>
            <p className="mt-1 text-sm text-slate-300">{c.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

