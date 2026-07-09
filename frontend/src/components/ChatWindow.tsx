import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

/** The scrolling message list. Purely presentational; all state lives in App. */
export function ChatWindow({
  messages,
  streaming,
}: {
  messages: ChatMessage[];
  streaming: boolean;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view as tokens stream in.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.length === 0 && (
        <div className="empty-state">
          <p className="empty-title">Ask about any YouTube video</p>
          <p>
            Open the video panel (🎬, top right) to ingest a video by URL, or
            just ask: the agent can search YouTube and ingest videos itself.
          </p>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageBubble
          key={i}
          message={msg}
          // The last assistant bubble shows a typing cursor while streaming.
          active={streaming && i === messages.length - 1}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
