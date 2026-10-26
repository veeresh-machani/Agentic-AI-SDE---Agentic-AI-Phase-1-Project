"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";

const API_BASE_URL = (
  process.env.NEXT_PUBLIC_HR_POLICY_API_URL || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

type UploadedDocument = {
  upload_id: string;
  filename: string;
  page_count: number;
  chunk_count: number;
};

type Source = {
  source: string;
  page: number;
  chunk_id?: number | null;
  snippet?: string;
};

type QueryResponse = {
  question: string;
  answer: string;
  query_type: string;
  routing_method: string;
  sources: Source[];
};

const DEFAULT_HR_DOC: UploadedDocument = {
  upload_id: "default",
  filename: "Company HR Policy Knowledge Base",
  page_count: 39,
  chunk_count: 124,
};

const SAMPLE_QUESTIONS = [
  "How many casual leaves are allocated on a half-yearly basis, and when are the cycles?",
  "What is the standard probation period and performance appraisal process?",
  "What are the entitlement details for maternity and paternity leave?",
  "What is the notice period required during probation versus after confirmation?",
];

export default function Home() {
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">(
    "checking"
  );

  const [activeDocument, setActiveDocument] =
    useState<UploadedDocument>(DEFAULT_HR_DOC);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);

  const [uploading, setUploading] = useState(false);
  const [loadingAnswer, setLoadingAnswer] = useState(false);

  const [uploadError, setUploadError] = useState("");
  const [queryError, setQueryError] = useState("");
  const [uploadMessage, setUploadMessage] = useState("");
  const [showUploadModal, setShowUploadModal] = useState(false);

  useEffect(() => {
    let cancelled = false;

    // Check backend health & fetch default document metadata
    fetch(`${API_BASE_URL}/health`, { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error("API health check failed");
        return res.json();
      })
      .then(() => {
        if (!cancelled) setApiStatus("online");
        return fetch(`${API_BASE_URL}/api/default-document`, {
          cache: "no-store",
        });
      })
      .then((res) => (res.ok ? res.json() : null))
      .then((doc) => {
        if (!cancelled && doc && doc.upload_id) {
          setActiveDocument((prev) =>
            prev.upload_id === "default" ? doc : prev
          );
        }
      })
      .catch(() => {
        if (!cancelled) setApiStatus("offline");
      });

    return () => {
      cancelled = true;
    };
  }, []);

  function handleFileChange(file: File | null) {
    setSelectedFile(file);
    setUploadError("");
    setUploadMessage("");
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!selectedFile) {
      setUploadError("Please choose a PDF file first.");
      return;
    }

    if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setUploadError("Only PDF files are supported.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setUploadError("The maximum PDF size is 10 MB.");
      return;
    }

    if (selectedFile.size === 0) {
      setUploadError("The selected file is empty.");
      return;
    }

    setUploading(true);
    setUploadError("");
    setUploadMessage("");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : `Upload failed with HTTP ${response.status}.`
        );
      }

      if (
        !data ||
        typeof data.upload_id !== "string" ||
        typeof data.chunk_count !== "number"
      ) {
        throw new Error("The API returned an unexpected upload response.");
      }

      const uploadedDoc = data as UploadedDocument;
      setActiveDocument(uploadedDoc);
      setUploadMessage(
        `"${uploadedDoc.filename}" indexed in memory successfully! Questions will now query this document.`
      );
      setSelectedFile(null);
      setResult(null);
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Could not upload the PDF."
      );
    } finally {
      setUploading(false);
    }
  }

  function handleResetToDefault() {
    setActiveDocument(DEFAULT_HR_DOC);
    setUploadMessage("");
    setUploadError("");
    setResult(null);
  }

  async function handleQuestion(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!question.trim()) {
      setQueryError("Please enter a question.");
      return;
    }

    setLoadingAnswer(true);
    setQueryError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          upload_id: activeDocument.upload_id,
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : `The request failed with HTTP ${response.status}.`
        );
      }

      if (!data || typeof data.answer !== "string") {
        throw new Error("The API returned an unexpected answer.");
      }

      setResult(data as QueryResponse);
    } catch (error) {
      setQueryError(
        error instanceof Error
          ? error.message
          : "Could not get an answer from the backend."
      );
    } finally {
      setLoadingAnswer(false);
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50 text-slate-900">
      <header className="border-b border-slate-200 bg-white/90 sticky top-0 z-10 backdrop-blur-sm">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-5 py-4 sm:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-blue-700 text-2xl text-white shadow-sm">
              📘
            </div>
            <div>
              <p className="text-lg font-bold tracking-tight text-slate-900">
                HR Policy Bot
              </p>
              <p className="text-xs text-slate-500">
                Instant answers with page citations
              </p>
            </div>
          </div>

          <div
            className="flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium shadow-sm"
            aria-live="polite"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                apiStatus === "online"
                  ? "bg-emerald-500 ring-2 ring-emerald-200"
                  : apiStatus === "offline"
                    ? "bg-red-500 ring-2 ring-red-200"
                    : "bg-amber-400 ring-2 ring-amber-200 animate-pulse"
              }`}
            />
            {apiStatus === "online"
              ? "API Connected"
              : apiStatus === "offline"
                ? "API Offline"
                : "Connecting..."}
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-4xl px-5 py-8 sm:px-8 sm:py-12">
        {/* Hero Section */}
        <section className="mb-8 text-center">
          <span className="mb-3 inline-flex items-center gap-1.5 rounded-full bg-blue-100 px-3.5 py-1 text-xs font-semibold uppercase tracking-wider text-blue-800">
            <span>✨</span> Zero Local Setup Needed
          </span>

          <h1 className="text-3xl font-extrabold tracking-tight sm:text-5xl text-slate-900">
            Ask Questions About HR Policies
          </h1>

          <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-600">
            Grounded answers synthesized with cross-encoder reranking and page citations.
            Pre-loaded with company policies, or upload your own PDF in memory.
          </p>
        </section>

        {/* Active Knowledge Base Banner */}
        <section className="mb-6 rounded-2xl border border-blue-200 bg-gradient-to-r from-blue-50 via-indigo-50 to-white p-4 sm:p-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div className="flex items-start sm:items-center gap-3">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-blue-600 text-lg text-white shadow-sm">
                📚
              </span>
              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-blue-700">
                  Active Knowledge Base
                </p>
                <p className="text-sm font-bold text-slate-900">
                  {activeDocument.filename}
                </p>
                <p className="text-xs text-slate-500">
                  {activeDocument.page_count} pages · {activeDocument.chunk_count} chunks indexed
                  {activeDocument.upload_id === "default"
                    ? " (Pre-loaded · No PDF upload required)"
                    : " (Session Upload · In-memory only)"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {activeDocument.upload_id !== "default" && (
                <button
                  type="button"
                  onClick={handleResetToDefault}
                  className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
                >
                  ↺ Reset to Company Policies
                </button>
              )}
              <button
                type="button"
                onClick={() => setShowUploadModal(!showUploadModal)}
                className="rounded-lg bg-blue-700 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-blue-800"
              >
                {showUploadModal ? "Hide Upload" : "Upload Custom PDF"}
              </button>
            </div>
          </div>

          {/* Optional Upload Form Dropdown/Drawer */}
          {showUploadModal && (
            <div className="mt-5 border-t border-blue-200/60 pt-5">
              <div className="mb-3">
                <h3 className="text-sm font-bold text-slate-800">
                  Upload a Custom PDF (Optional)
                </h3>
                <p className="text-xs text-slate-500">
                  Want to ask questions about a different handbook or PDF?
                  The document is processed entirely in memory—no files are saved to disk.
                </p>
              </div>

              <form onSubmit={handleUpload}>
                <label
                  htmlFor="pdf-upload"
                  className="flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed border-blue-300 bg-white px-4 py-6 text-center transition hover:border-blue-500 hover:bg-blue-50/50"
                >
                  <span className="mb-1 text-2xl">📄</span>
                  <span className="text-xs font-semibold text-slate-800">
                    {selectedFile ? selectedFile.name : "Choose a PDF from your computer"}
                  </span>
                  <span className="text-[11px] text-slate-400">
                    {selectedFile
                      ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB`
                      : "Up to 10 MB · Processed strictly in-memory"}
                  </span>
                  <input
                    id="pdf-upload"
                    type="file"
                    accept=".pdf,application/pdf"
                    className="sr-only"
                    onChange={(e) => handleFileChange(e.target.files?.[0] ?? null)}
                  />
                </label>

                <div className="mt-3 flex items-center justify-end gap-2">
                  <button
                    type="button"
                    onClick={() => setShowUploadModal(false)}
                    className="rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={uploading || !selectedFile}
                    className="flex items-center gap-1.5 rounded-lg bg-blue-700 px-4 py-1.5 text-xs font-semibold text-white transition hover:bg-blue-800 disabled:opacity-50"
                  >
                    {uploading ? "Indexing in memory..." : "Index & Switch to PDF"}
                  </button>
                </div>
              </form>

              {uploadError && (
                <p className="mt-3 rounded-lg border border-red-200 bg-red-50 p-2 text-xs text-red-700">
                  {uploadError}
                </p>
              )}

              {uploadMessage && (
                <p className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 p-2 text-xs text-emerald-800">
                  ✓ {uploadMessage}
                </p>
              )}
            </div>
          )}
        </section>

        {/* Main Question Box */}
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xl shadow-slate-200/50 sm:p-7">
          <div className="mb-4">
            <h2 className="text-lg font-bold text-slate-900">
              Ask your question
            </h2>
            <p className="text-xs text-slate-500">
              Questions are answered using only verified citations from{" "}
              <span className="font-semibold text-slate-700">{activeDocument.filename}</span>.
            </p>
          </div>

          {/* Quick sample question chips */}
          <div className="mb-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Suggested questions:
            </p>
            <div className="flex flex-wrap gap-2">
              {SAMPLE_QUESTIONS.map((sampleQ, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setQuestion(sampleQ)}
                  className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5 text-left text-xs font-medium text-slate-700 transition hover:border-blue-300 hover:bg-blue-50/70 hover:text-blue-800"
                >
                  💬 {sampleQ}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleQuestion}>
            <label htmlFor="question" className="sr-only">
              Your question
            </label>

            <textarea
              id="question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. How many casual leaves can I take in the first half of the year?"
              maxLength={2000}
              rows={3}
              disabled={loadingAnswer}
              className="w-full resize-y rounded-xl border border-slate-300 bg-white px-4 py-3 text-sm leading-6 outline-none transition placeholder:text-slate-400 focus:border-blue-600 focus:ring-4 focus:ring-blue-100 disabled:bg-slate-50"
            />

            <div className="mt-2 flex items-center justify-between text-xs text-slate-400">
              <span>Grounding: FAISS vector retrieval + cross-encoder reranking</span>
              <span>{question.length}/2000</span>
            </div>

            <button
              type="submit"
              disabled={loadingAnswer || !question.trim()}
              className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-700 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loadingAnswer ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                  Retrieving, reranking, and generating answer...
                </>
              ) : (
                <>
                  Ask HR Policy Bot <span aria-hidden="true">→</span>
                </>
              )}
            </button>
          </form>

          {queryError && (
            <div className="mt-5 rounded-xl border border-amber-300 bg-amber-50 p-4" role="alert">
              <p className="font-semibold text-amber-900 text-sm">
                Could not generate an answer
              </p>
              <p className="mt-1 text-xs text-amber-800">{queryError}</p>
            </div>
          )}
        </section>

        {/* Answer and Citations Card */}
        {result && (
          <section className="mt-8 space-y-5" aria-live="polite">
            <div className="rounded-2xl border border-emerald-200 bg-white p-5 shadow-sm sm:p-7">
              <div className="mb-4 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-emerald-100 text-sm font-bold text-emerald-800">
                    ✓
                  </span>
                  <h2 className="text-lg font-bold text-slate-900">Answer</h2>
                </div>

                <div className="flex items-center gap-2 text-xs">
                  <span className="rounded-md bg-slate-100 px-2 py-1 text-slate-600 font-medium">
                    Type: {result.query_type}
                  </span>
                  <span className="rounded-md bg-blue-50 px-2 py-1 text-blue-700 font-medium">
                    Router: {result.routing_method}
                  </span>
                </div>
              </div>

              <div className="whitespace-pre-wrap text-sm leading-7 text-slate-800">
                {result.answer?.trim() ? (
                  result.answer
                ) : (
                  <span className="italic text-slate-400">
                    No answer text was returned. Please review the retrieved source citations below.
                  </span>
                )}
              </div>
            </div>

            {/* Sources / Evidence */}
            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm sm:p-7">
              <details open>
                <summary className="cursor-pointer text-base font-bold text-slate-900">
                  Citations & Sources ({result.sources?.length ?? 0})
                </summary>

                <p className="mt-2 text-xs leading-5 text-slate-500">
                  These verified chunks were retrieved, re-scored by the cross-encoder, and provided to the LLM as grounding context.
                </p>

                <div className="mt-4 space-y-3">
                  {result.sources?.length ? (
                    result.sources.map((src, index) => {
                      const filename =
                        src.source?.split(/[\\/]/).pop() ||
                        activeDocument.filename;

                      return (
                        <article
                          key={`${src.source}-${src.chunk_id ?? index}-${index}`}
                          className="rounded-xl border border-slate-200 bg-slate-50/50 p-4 transition hover:bg-slate-50"
                        >
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded-md bg-blue-100 px-2.5 py-0.5 text-xs font-bold text-blue-800">
                              Page {src.page}
                            </span>
                            <span className="text-xs font-semibold text-slate-800 break-all">
                              {filename}
                            </span>
                            {src.chunk_id != null && (
                              <span className="text-[11px] text-slate-400">
                                (Chunk #{src.chunk_id})
                              </span>
                            )}
                          </div>

                          {src.snippet && (
                            <p className="mt-2.5 whitespace-pre-wrap text-xs leading-6 text-slate-600 font-mono bg-white p-3 rounded-lg border border-slate-200/60">
                              {src.snippet}
                            </p>
                          )}
                        </article>
                      );
                    })
                  ) : (
                    <p className="text-xs text-slate-500">No source citations returned.</p>
                  )}
                </div>
              </details>
            </div>
          </section>
        )}

        <footer className="mt-12 text-center text-xs text-slate-400">
          <p>HR Policy Assistant · Zero local PDF saving required · In-memory processing</p>
          <p className="mt-1">Next.js · FastAPI · FAISS · LangChain LCEL · Cross-Encoder Reranking</p>
        </footer>
      </div>
    </main>
  );
}