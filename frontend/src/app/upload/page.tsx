"use client";

import React, { useState, DragEvent } from "react";
import { uploadPdf } from "@/lib/api";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";

export default function UploadPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  /** Handle File Selection */
  const handleFileSelect = (files: FileList | null) => {
    const selected = files?.[0];
    if (selected && selected.type === "application/pdf") {
      setFile(selected);
      setStatus(`Selected: ${selected.name}`);
    } else {
      setStatus("❌ Please upload a valid PDF file.");
    }
  };

  /** Drag Events */
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };

  /** Upload Function */
  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return setStatus("⚠️ No file selected.");

    setBusy(true);
    setStatus("🔄 Uploading and processing PDF...");

    try {
      const res = await uploadPdf(file);
      if (res.status === "success" && res.doc_id) {
        setStatus(`✅ Processed ${res.chunks} chunks. Redirecting...`);
        setTimeout(() => router.push(`/chat/${res.doc_id}`), 1200);
      } else {
        setStatus(res.message || "Upload failed.");
      }
    } catch (err: any) {
      setStatus("❌ Error while uploading. Try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="max-w-3xl mx-auto grid gap-6 p-4">
      {/* Title */}
      <motion.h1
        className="text-3xl font-bold text-center"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        📄 Upload a PDF to Begin
      </motion.h1>

      {/* Drag & Drop Card */}
      <motion.div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
          isDragging ? "border-sky-500 bg-sky-900/20" : "border-slate-700 bg-slate-900/40"
        }`}
        onClick={() => document.getElementById("pdfInput")?.click()}
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.02 }}
      >
        <input
          id="pdfInput"
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={(e) => handleFileSelect(e.target.files)}
        />
        {!file ? (
          <p className="text-slate-300">
            Drag & drop your PDF here, or <span className="text-sky-400 font-semibold">click to select</span>
          </p>
        ) : (
          <p className="text-sky-400 font-semibold">📄 {file.name}</p>
        )}
      </motion.div>

      {/* Process button */}
      <div className="text-center">
        <button
          onClick={handleUpload}
          disabled={!file || busy}
          className="bg-sky-500 hover:bg-sky-600 disabled:opacity-50 text-white font-semibold px-6 py-3 rounded-lg transition-all"
        >
          {busy ? "⏳ Processing..." : "🚀 Start Processing"}
        </button>
      </div>

      {/* Status Message */}
      {status && (
        <motion.div
          className="text-center text-sm text-slate-300 bg-slate-800/50 p-3 rounded-lg border border-slate-700"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {status}
        </motion.div>
      )}
    </section>
  );
}

