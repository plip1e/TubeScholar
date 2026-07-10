import type { ChatMessage } from "../types";

/** One chat bubble. The blinking cursor (▍) marks the answer still streaming in. */
export function MessageBubble({
  message,
  active,
}: {
  message: ChatMessage;
  active: boolean;
}) {
  const classes = ["bubble", message.role, message.error ? "error" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={`bubble-row ${message.role}`}>
      <div className={classes}>
        {message.content}
        {active && !message.content && message.status && (
          <span className="status-line">{message.status}</span>
        )}
        {active && <span className="cursor">▍</span>}
      </div>
    </div>
  );
}
