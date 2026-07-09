import { useState, type FormEvent } from "react";
import type { IngestResult, VideoInfo } from "../types";

/** Turn an ISO-8601 duration ("PT1H2M3S") into "1:02:03". */
function formatDuration(iso: string | null): string {
  if (!iso) return "";
  const match = iso.match(/PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?/);
  if (!match) return "";
  const [, h, m, s] = match;
  const parts = [h, m ?? "0", s ?? "0"].filter((p) => p !== undefined);
  return parts
    .map((p, i) => (i === 0 ? String(Number(p)) : String(Number(p)).padStart(2, "0")))
    .join(":");
}

/** "1234567" -> "1.2M views" */
function formatViews(count: string | null): string {
  if (!count) return "";
  const n = Number(count);
  if (Number.isNaN(n)) return "";
  return `${Intl.NumberFormat("en", { notation: "compact" }).format(n)} views`;
}

/** Human-readable line for an ingest status dict. */
function statusMessage(result: IngestResult): string {
  switch (result.status) {
    case "ingested":
      return `Ingested (${result.chunks} chunks).`;
    case "skipped":
      return "Already ingested and up to date.";
    case "invalid_url":
      return "That doesn't look like a YouTube URL.";
    case "not_found":
      return "Video not found — it may be private or deleted.";
    case "transcript_unavailable":
      return "No transcript available for this video.";
    default:
      return result.reason ?? `Ingestion failed (${result.status}).`;
  }
}

/** Sidebar (right edge): the ingested-video library + a form to add more.
 *
 * Collapses to a thin rail. Both branches below return an <aside> in the same
 * position, so React reuses the DOM node and the CSS width transition runs on
 * toggle — and this component never unmounts, so the form state (typed URL,
 * ingest-in-progress) survives collapsing. */
export function VideoPanel({
  videos,
  onIngest,
  backendDown,
  open,
  onToggle,
}: {
  videos: VideoInfo[];
  onIngest: (url: string) => Promise<IngestResult>;
  backendDown: boolean;
  open: boolean;
  onToggle: () => void;
}) {
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<{ text: string; ok: boolean } | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = url.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setStatus({ text: "Ingesting… (fetching transcript + embedding)", ok: true });
    try {
      const result = await onIngest(trimmed);
      const ok = result.status === "ingested" || result.status === "skipped";
      setStatus({ text: statusMessage(result), ok });
      if (ok) setUrl("");
    } catch (err) {
      setStatus({
        text: err instanceof Error ? err.message : String(err),
        ok: false,
      });
    } finally {
      setBusy(false);
    }
  }

  if (!open) {
    // The whole rail is one big button — easiest possible target to reopen.
    return (
      <aside
        className="video-panel collapsed"
        onClick={onToggle}
        role="button"
        aria-expanded={false}
        aria-label={`Show videos (${videos.length})`}
        title="Show videos"
      >
        <span className="rail-chevron">◀</span>
        <span className="rail-icon">🎬</span>
        <span className="rail-count">{videos.length}</span>
        {backendDown && (
          <span className="rail-warn" title="Backend unreachable">
            ⚠
          </span>
        )}
      </aside>
    );
  }

  return (
    <aside className="video-panel">
      <div className="panel-header">
        <h2>Videos</h2>
        <button
          className="panel-toggle"
          onClick={onToggle}
          aria-expanded={true}
          aria-label="Hide videos"
          title="Hide videos"
        >
          ▶
        </button>
      </div>

      <form className="ingest-form" onSubmit={handleSubmit}>
        <input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="Paste a YouTube URL…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !url.trim()}>
          {busy ? "Ingesting…" : "Ingest"}
        </button>
      </form>
      {status && (
        <p className={`ingest-status ${status.ok ? "ok" : "fail"}`}>{status.text}</p>
      )}

      {backendDown && (
        <p className="ingest-status fail">
          Can't reach the backend — is uvicorn running on :8000?
        </p>
      )}

      <ul className="video-list">
        {videos.length === 0 && !backendDown && (
          <li className="video-empty">Nothing ingested yet.</li>
        )}
        {videos.map((v) => (
          <li key={v.video_id} className="video-item">
            <a href={v.url} target="_blank" rel="noreferrer" title={v.title ?? v.video_id}>
              {v.title ?? v.video_id}
            </a>
            <span className="video-meta">
              {[v.creator, formatDuration(v.duration), formatViews(v.view_count)]
                .filter(Boolean)
                .join(" · ")}
            </span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
