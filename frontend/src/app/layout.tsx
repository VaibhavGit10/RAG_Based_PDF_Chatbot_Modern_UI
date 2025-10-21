import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "PDF Insight AI",
  description: "RAG-based PDF Q&A with Qdrant, ONNX embeddings, and Groq",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-screen bg-slate-950 text-slate-100 antialiased">
        {/* Header */}
        <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/70 backdrop-blur">
          <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-3">
            <Link href="/" className="group flex items-center gap-3">
              <Image
                src="/logo.svg"
                alt="PDF Insight AI logo"
                width={36}
                height={36}
                priority
                className="transition-transform group-hover:scale-105"
              />
              <span className="text-lg font-semibold tracking-wide">PDF Insight AI</span>
            </Link>
            <nav className="ml-auto flex items-center gap-4 text-sm text-slate-300">
              <Link href="/upload" className="hover:text-white">Upload</Link>
              <Link href="/settings" className="hover:text-white">Settings</Link>
              <a
                href="https://github.com/"
                target="_blank"
                rel="noreferrer"
                className="rounded-md border border-slate-800 px-2 py-1 hover:border-slate-700 hover:text-white"
              >
                GitHub
              </a>
            </nav>
          </div>
        </header>

        {/* Main */}
        <main className="mx-auto w-full max-w-6xl px-4 py-6">{children}</main>

        {/* Footer */}
        <footer className="border-t border-slate-800 py-6 text-center text-xs text-slate-400">
          © {new Date().getFullYear()} PDF Insight AI • Made for delightful document conversations
        </footer>
      </body>
    </html>
  );
}
