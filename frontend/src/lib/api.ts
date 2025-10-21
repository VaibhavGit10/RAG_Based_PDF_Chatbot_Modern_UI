// /src/lib/api.ts

// ✅ Ensure no trailing slash in backend URL
const BASE_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type UploadResponse = {
  status: "success" | "error";
  doc_id?: string;       // Unique document ID (filename + UUID)
  chunks?: number;       // Number of chunks stored in vector DB
  message?: string;      // Message for UI display
};

export type AskResponse = {
  status: "success" | "error";
  answer?: string;       // AI-generated answer
  sources?: { text?: string }[] | string[]; // Retrieved context for transparency
  message?: string;      // Error message if failure
};

/**
 * 📄 Upload a PDF to the backend and receive a doc_id
 */
export async function uploadPdf(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${BASE_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      return { status: "error", message: `Upload failed (${res.status})` };
    }

    return await res.json();
  } catch (err: any) {
    return { status: "error", message: err.message || "Network error during upload" };
  }
}

/**
 * 💬 Ask a question using the given doc_id
 */
export async function askQuestion(
  question: string,
  docId: string,
  topK = 4
): Promise<AskResponse> {
  if (!docId) {
    return { status: "error", message: "Missing document ID" };
  }

  try {
    const res = await fetch(`${BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, doc_id: docId, top_k: topK }),
    });

    if (!res.ok) {
      return { status: "error", message: `Ask failed (${res.status})` };
    }

    return await res.json();
  } catch (err: any) {
    return { status: "error", message: err.message || "Network error during ask" };
  }
}

/**
 * 🔍 Backend Health Check
 */
export async function health(): Promise<{ status: string }> {
  try {
    const res = await fetch(`${BASE_URL}/health`, { cache: "no-store" });
    return await res.json();
  } catch (err) {
    return { status: "error" };
  }
}

/**
 * 🚀 Streaming (will be implemented next)
 * - This is a placeholder function we will enhance based on your backend support
 */
export async function askQuestionStream(
  question: string,
  docId: string,
  onToken: (token: string) => void
): Promise<void> {
  // To be implemented in next step once backend supports streaming
  console.warn("Streaming is not implemented yet.");
}

