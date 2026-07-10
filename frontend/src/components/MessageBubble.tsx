import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ChatMessage } from "../types";

/** One chat bubble. The blinking cursor (▍) marks the answer still streaming in.
 *
 * Assistant text is rendered as markdown (the model answers in it: bold,
 * lists, tables). react-markdown builds real React elements, so there is no
 * innerHTML and no script-injection surface; remark-gfm adds GitHub-flavored
 * extras like tables. User text stays literal: people expect what they typed,
 * exactly as typed.
 */
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
        {message.role === "assistant" && !message.error ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {message.content}
          </ReactMarkdown>
        ) : (
          message.content
        )}
        {active && !message.content && message.status && (
          <span className="status-line">{message.status}</span>
        )}
        {active && <span className="cursor">▍</span>}
      </div>
    </div>
  );
}
