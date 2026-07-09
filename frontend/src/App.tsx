import { useCallback, useEffect, useState } from "react";
import { ingestVideo, listVideos, streamChat } from "./api/client";
import type { ChatMessage, IngestResult, VideoInfo } from "./types";
import { ChatWindow } from "./components/ChatWindow";
import { Composer } from "./components/Composer";
import { VideoPanel } from "./components/VideoPanel";

/** Root component. It owns all the state; children just render it.
 *
 * This "state lives at the top, flows down as props, events flow back up as
 * callbacks" shape is the standard React pattern; when the app grows past
 * what that handles comfortably, that's the cue to reach for a state library.
 */
export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  // The LangGraph checkpoint key: one thread = one conversation memory.
  // A new id per page load (or "New chat") starts a fresh conversation.
  const [threadId, setThreadId] = useState(() => crypto.randomUUID());
  const [videos, setVideos] = useState<VideoInfo[]>([]);
  const [backendDown, setBackendDown] = useState(false);
  // Video panel visibility, collapsed to a thin rail by default.
  const [panelOpen, setPanelOpen] = useState(false);

  const refreshVideos = useCallback(async () => {
    try {
      setVideos(await listVideos());
      setBackendDown(false);
    } catch {
      setBackendDown(true);
    }
  }, []);

  // Load the sidebar once on mount.
  useEffect(() => {
    refreshVideos();
  }, [refreshVideos]);

  async function send(text: string) {
    // Two updates in one: the user's message plus an empty assistant bubble
    // that the stream below will fill token by token.
    setMessages((m) => [
      ...m,
      { role: "user", content: text },
      { role: "assistant", content: "" },
    ]);
    setStreaming(true);
    try {
      for await (const token of streamChat(text, threadId)) {
        // Functional update: React batches state changes, so we must derive
        // from the latest state (m), not from a stale `messages` closure.
        setMessages((m) => {
          const next = m.slice();
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, content: last.content + token };
          return next;
        });
      }
    } catch (err) {
      setMessages((m) => {
        const next = m.slice();
        next[next.length - 1] = {
          role: "assistant",
          content: `Something went wrong: ${err instanceof Error ? err.message : err}`,
          error: true,
        };
        return next;
      });
    } finally {
      setStreaming(false);
      // The agent can ingest videos mid-conversation, so keep the sidebar honest.
      refreshVideos();
    }
  }

  async function ingest(url: string): Promise<IngestResult> {
    const result = await ingestVideo(url);
    if (result.status === "ingested" || result.status === "skipped") {
      refreshVideos();
    }
    return result;
  }

  function newChat() {
    setThreadId(crypto.randomUUID());
    setMessages([]);
  }

  return (
    <div className="app">
      <main className="chat-area">
        <header className="chat-header">
          <h1>
            Tube<span className="accent">Scholar</span>
          </h1>
          <button className="ghost-button" onClick={newChat} disabled={streaming}>
            + New chat
          </button>
        </header>
        <ChatWindow messages={messages} streaming={streaming} />
        <Composer onSend={send} disabled={streaming} />
      </main>
      <VideoPanel
        videos={videos}
        onIngest={ingest}
        backendDown={backendDown}
        open={panelOpen}
        onToggle={() => setPanelOpen((o) => !o)}
      />
    </div>
  );
}
