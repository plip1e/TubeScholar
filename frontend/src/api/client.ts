/** All communication with the FastAPI backend lives here.
 *
 * Every call goes to an /api/... path. In dev, Vite's proxy (vite.config.ts)
 * forwards those to the backend on :8000.
 *
 * The interesting one is streamChat(): the backend streams the answer as
 * Server-Sent Events (SSE). The browser's built-in EventSource class only
 * supports GET, and our endpoint is a POST, so we read the response body
 * ourselves with fetch() + ReadableStream and parse the SSE wire format by
 * hand. It's ~20 lines and worth understanding, because this is all SSE is:
 * a long-lived HTTP response whose body arrives in pieces.
 */

import type { ChatEvent, IngestResult, VideoInfo } from "../types";

/** Stream the assistant's turn as a series of typed events.
 *
 * Usage:  for await (const ev of streamChat(text, threadId)) { ... }
 *
 * An async generator: each `yield` hands the caller one event as soon as it
 * arrives over the wire, and the function stays suspended in between. The
 * SSE wire format is text events separated by a blank line, each shaped like:
 *
 *     data: {"status": "..."}   -> {type: "status"}  what the agent is doing
 *     data: {"token": "..."}    -> {type: "token"}   a piece of the answer
 *     data: {"reset": true}     -> {type: "reset"}   discard shown text, a
 *                                                    revised answer follows
 *
 * with a final `data: {"done": true}` event when the answer is complete.
 */
export async function* streamChat(
  message: string,
  threadId: string,
): AsyncGenerator<ChatEvent> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // Field names must match the backend's ChatRequest Pydantic model.
    body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!res.ok || !res.body) {
    throw new Error(`Chat request failed (HTTP ${res.status})`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  // Network chunks split anywhere, even mid-character, so we accumulate into
  // a buffer and only consume complete events (terminated by a blank line).
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() ?? ""; // last piece may be incomplete, keep it

    for (const event of events) {
      for (const line of event.split(/\r?\n/)) {
        if (!line.startsWith("data:")) continue; // skip SSE comments/pings
        const payload = JSON.parse(line.slice("data:".length).trim());
        if (payload.done) return;
        if (payload.token) yield { type: "token", text: payload.token };
        else if (payload.status) yield { type: "status", text: payload.status };
        else if (payload.reset) yield { type: "reset" };
      }
    }
  }
}

/** Ask the backend to ingest a YouTube video. Resolves to the pipeline's
 * status dict either way; check `status`, don't try/catch for app errors. */
export async function ingestVideo(url: string): Promise<IngestResult> {
  const res = await fetch("/api/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(`Ingest request failed (HTTP ${res.status})`);
  return res.json();
}

/** Everything currently in the vector store, for the sidebar. */
export async function listVideos(): Promise<VideoInfo[]> {
  const res = await fetch("/api/videos");
  if (!res.ok) throw new Error(`Video list failed (HTTP ${res.status})`);
  return res.json();
}
