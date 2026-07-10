/** Shared shapes for talking to the backend.
 *
 * These mirror the backend's Pydantic models / dicts by hand. TypeScript can't
 * see Python, so if a backend field changes, this file must change with it:
 * one place to look, same idea as Pydantic being the single source of truth
 * on the other side.
 */

/** One turn in the conversation, as rendered in the chat window. */
export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  /** Set when the request itself failed (network/server), styled differently. */
  error?: boolean;
  /** What the agent is doing right now (from a status event); shown while
   * waiting for answer tokens. */
  status?: string;
}

/** One event from the /api/chat SSE stream, as parsed by streamChat(). */
export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "status"; text: string }
  | { type: "reset" };

/** What GET /videos returns per video; mirrors VideoList.as_dict() in func.py. */
export interface VideoInfo {
  lst_placement: number;
  video_id: string;
  url: string;
  creator: string | null;
  creator_description: string | null;
  title: string | null;
  channel_id: string | null;
  published_at: string | null;
  view_count: string | null;
  like_count: string | null;
  /** ISO-8601 duration, e.g. "PT12M34S". */
  duration: string | null;
}

/** What POST /ingest returns: the pipeline's status dict.
 * `status` is "ingested" | "skipped" on success, or a named failure
 * ("invalid_url", "not_found", "transcript_unavailable", "error", ...). */
export interface IngestResult {
  status: string;
  video_id?: string;
  chunks?: number;
  reason?: string;
  action?: string;
}
