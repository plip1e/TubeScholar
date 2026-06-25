'''This module holds the agent that ingests videos using the given URL, vectorises them, and adds them to the Chromadb database'''

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from requests.exceptions import RequestException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from urllib.parse import urlparse, parse_qs
from langchain_core.tools import tool
from langsmith import traceable
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from langchain_chroma import Chroma
import chromadb, yt_dlp

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

from func import VideoList


load_dotenv()
google_api_key = os.getenv("PAID_GEMINI_API") # FREE_GEMINI_API
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
WS_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME")
WS_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD")
# Local Whisper speech-to-text is gated off by default while audio transcription
# is being offloaded to an external Colab + FastAPI service (avoids loading the
# model into memory on every startup). Set WHISPER_ENABLED=1 to re-enable it.
WHISPER_ENABLED = os.getenv("WHISPER_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


class TranscriptUnavailable(Exception):
    """No transcript could be produced (no captions, and Whisper is unavailable).

    Carried up to ingest_video so it can return an agent-readable status with the
    metadata the agent should fall back on, instead of crashing the run.
    """


def _youtube_error(e: HttpError) -> dict:
    """Turn a googleapiclient HttpError into a flat, agent-readable dict.

    Pulls the HTTP status and the API's own reason/message out of the error so a
    tool can return something the supervisor can act on (e.g. quota exceeded vs.
    the API/key being blocked) instead of crashing the run.
    """
    status = getattr(getattr(e, "resp", None), "status", None)
    reason = ""
    message = str(e)
    try:
        details = (e.error_details or [{}])[0] if hasattr(e, "error_details") else {}
        reason = details.get("reason", "")
        message = details.get("message", message)
    except Exception:
        pass

    if status == 403 and ("blocked" in message.lower() or reason == "forbidden"):
        action = ("The YouTube Data API request was blocked. Check that the API key is valid, "
                  "that the YouTube Data API v3 is enabled for its project, and that any API/IP/"
                  "referrer restrictions on the key allow this call. Do not retry until the key is fixed.")
    elif status == 403 and reason in ("quotaExceeded", "rateLimitExceeded"):
        action = "The YouTube Data API quota/rate limit was hit. Wait and retry later, or ask the user for a fresh quota."
    elif status == 400:
        action = "The YouTube Data API rejected the request as malformed. Check the query/parameters."
    elif status == 404:
        action = ("The YouTube Data API returned 404 (not found). The requested resource/endpoint "
                  "could not be located — check the method and that the YouTube Data API v3 is enabled "
                  "for the key's project. Do not retry the identical request.")
    else:
        action = "The YouTube Data API call failed. Do not retry the identical request immediately."

    return {
        "status": "youtube_api_error",
        "http_status": status,
        "reason": reason or "unknown",
        "message": message,
        "action": action,
    }


class VideoIngestionPipeline:
    def __init__(self, google_api_key: str, youtube_api_key: str):
        self.max_age_days = 7
        self._youtube_api_key = youtube_api_key
        # The googleapiclient (httplib2) and youtube_transcript_api (requests
        # session) clients are NOT thread-safe — sharing one across the ingestion
        # thread pool interleaves their TLS streams (SSL: WRONG_VERSION_NUMBER).
        # So each thread lazily builds its own via the youtube/ytt_api properties.
        self._thread_local = threading.local()
        # Disabled by default (see WHISPER_ENABLED)
        self.whisper_model = None
        if WHISPER_ENABLED:
            import whisper
            self.whisper_model = whisper.load_model("base")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        self.vectorstore = Chroma(
            collection_name="youtube_videos",
            embedding_function=self.embeddings,
            # anchored to app/data so the same DB is used
            persist_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")
        )

        # quick-access metadata registry
        self.videos: dict[int, VideoList] = {}
        self._placement_by_id: dict[str, int] = {}
        self._channel_desc_cache: dict[str, str] = {}
        # serializes the Chroma write + registry/placement mutation so concurrent
        # ingestion workers (see ingest_video's thread pool) can't race on them.
        self._write_lock = threading.Lock()
        self._rebuild_registry()

    # -- per-thread clients ---------------------------------------------------------
    # Built lazily, one set per thread, so the ingestion thread pool never shares a
    # non-thread-safe HTTP transport. Every call site uses self.youtube/self.ytt_api
    # unchanged.

    @property
    def youtube(self):
        client = getattr(self._thread_local, "youtube", None)
        if client is None:
            client = discovery.build("youtube", "v3", developerKey=self._youtube_api_key)
            self._thread_local.youtube = client
        return client

    @property
    def ytt_api(self):
        api = getattr(self._thread_local, "ytt_api", None)
        if api is None:
            api = YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
                proxy_username=WS_USERNAME,
                proxy_password=WS_PASSWORD,
            ))
            self._thread_local.ytt_api = api
        return api

    # -- private helpers ------------------------------------------------------------

    def _get_video_id(self, url: str) -> str:
        if not url or not isinstance(url, str):
            raise ValueError("No URL provided.")
        if "youtu.be" in url:
            vid = url.split("/")[-1].split("?")[0]
        elif "/shorts/" in url:
            vid = url.split("/shorts/")[-1].split("?")[0]
        else:
            vid = (parse_qs(urlparse(url).query).get("v") or [None])[0]
        if not vid:
            raise ValueError(f"Could not extract a video id from URL: {url!r}")
        return vid

    def _get_channel_description(self, channel_id: str) -> str:
        """Channel ('creator') description, cached by channel_id."""
        if channel_id in self._channel_desc_cache:
            return self._channel_desc_cache[channel_id]
        try:
            response = self.youtube.channels().list(
                part="snippet",
                id=channel_id
            ).execute()
            items = response.get("items") or []
            description = items[0]["snippet"]["description"] if items else ""
        except (HttpError, KeyError, IndexError) as e:
            # description is optional metadata; never let it sink an ingestion
            print(f"[channel] could not fetch description for {channel_id}: {type(e).__name__}: {e}")
            description = ""
        self._channel_desc_cache[channel_id] = description
        return description

    def _get_metadata(self, video_id: str) -> dict:
        response = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()
        items = response.get("items") or []
        if not items:
            # empty when the id is invalid, or the video is private/deleted/region-blocked
            return None
        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content_details = item.get("contentDetails", {})
        channel_id = snippet.get("channelId")
        return {
            "video_id":    video_id,
            "url":         f"https://www.youtube.com/watch?v={video_id}",
            "title":       snippet.get("title"),
            "description": snippet.get("description"),
            "channel":     snippet.get("channelTitle"),
            "channel_id":  channel_id,
            "channel_description": self._get_channel_description(channel_id) if channel_id else "",
            "published_at": snippet.get("publishedAt"),
            "view_count":  stats.get("viewCount"),
            "like_count":  stats.get("likeCount"),
            "duration":    content_details.get("duration"),
        }

    def _get_top_comments(self, video_id: str, max_comments: int = 15) -> list[dict]:
        """Top-level comments by relevance — fallback overview material.

        Used when a video has no usable transcript so the agent has the audience's
        own words to infer what the video is about. Returns [] (never raises) if
        comments are disabled, missing, or the call fails — it's optional context.
        """
        try:
            response = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                order="relevance",
                maxResults=max_comments,
                textFormat="plainText",
            ).execute()
        except HttpError as e:
            status = getattr(getattr(e, "resp", None), "status", None)
            print(f"[comments] could not fetch comments for {video_id} (HTTP {status}); likely disabled")
            return []
        comments = []
        for item in response.get("items") or []:
            top = (((item.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {})
            text = top.get("textDisplay")
            if text:
                comments.append({
                    "author": top.get("authorDisplayName"),
                    "text": text,
                    "like_count": top.get("likeCount"),
                })
        return comments

    def _get_transcript(self, video_id: str) -> dict:
        try:
            fetched = self.ytt_api.fetch(video_id)
            return {
                "transcript": " ".join([s.text for s in fetched.snippets]),
                "transcript_source": "captions"
            }
        except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, RequestException) as e:
            if not (WHISPER_ENABLED and self.whisper_model is not None):
                # Whisper fallback is currently disabled
                raise TranscriptUnavailable(
                    f"No captions/transcript for {video_id} ({type(e).__name__}) and Whisper "
                    f"speech-to-text is currently unavailable."
                ) from e
            # Whisper enabled: download the audio and transcribe it locally
            print(f"[transcript] captions unavailable for {video_id} ({type(e).__name__}); falling back to Whisper")
            try:
                return {
                    "transcript": self._whisper_transcribe(video_id),
                    "transcript_source": "whisper"
                }
            except Exception as we:
                # Whisper fallback failed too (download blocked, no audio, ffmpeg
                # missing, etc.). Surface it to the caller rather than crashing.
                raise TranscriptUnavailable(
                    f"Could not obtain a transcript for {video_id}: captions unavailable "
                    f"({type(e).__name__}) and Whisper fallback failed ({type(we).__name__}: {we})."
                ) from we

    def _whisper_transcribe(self, video_id: str) -> str:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": video_id,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True
        }
        audio_path = f"{video_id}.mp3"
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            result = self.whisper_model.transcribe(audio_path)
            return result["text"]
        finally:
            # always clean up the temp audio, even if transcription raised
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError as e:
                    print(f"[whisper] could not remove temp file {audio_path}: {e}")

    def _chunk_transcript(self, transcript: str, chunk_size=500, overlap=50) -> list[str]:
        words = transcript.split()
        return [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
            if words[i:i + chunk_size]
        ]

    def _is_stale(self, video_id: str) -> bool:
        results = self.vectorstore.get(where={"video_id": video_id}, limit=1)
        if not results["ids"]:
            return True  # not in store yet
        fetched_at = results["metadatas"][0]["fetched_at"]
        age = datetime.now() - datetime.fromisoformat(fetched_at)
        return age.days > self.max_age_days

    def _register_video(self, meta: dict) -> VideoList:
        """Add or update a video in the quick-access registry.

        Placement is stable: a known video_id keeps its original slot on
        re-ingestion, a new one is appended at the next 1-based position.
        """
        video_id = meta["video_id"]
        if video_id in self._placement_by_id:
            placement = self._placement_by_id[video_id]
        else:
            placement = len(self.videos) + 1
            self._placement_by_id[video_id] = placement

        entry = VideoList(
            lst_placement=placement,
            video_id=video_id,
            url=meta.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            creator=meta.get("channel"),
            creator_description=meta.get("channel_description"),
            title=meta.get("title"),
            channel_id=meta.get("channel_id"),
            published_at=meta.get("published_at"),
            view_count=meta.get("view_count"),
            like_count=meta.get("like_count"),
            duration=meta.get("duration"),
        )
        self.videos[placement] = entry
        return entry

    def _rebuild_registry(self) -> None:
        """Repopulate the registry from the persisted vector store on startup.

        Chunk metadata holds everything except the transcript, so previously
        ingested videos are available to agent tools right after a restart.
        Ordered by fetched_at so placements stay stable across runs.
        """
        self.videos = {}
        self._placement_by_id = {}

        try:
            stored = self.vectorstore.get()
            metadatas = stored.get("metadatas") or []
        except Exception as e:
            # a corrupt/unreadable store shouldn't prevent the pipeline from
            # starting — begin with an empty registry instead.
            print(f"[registry] could not load persisted store: {type(e).__name__}: {e}")
            return

        # collapse chunk metadata down to one entry per video
        per_video: dict[str, dict] = {}
        for meta in metadatas:
            video_id = meta.get("video_id")
            if video_id and video_id not in per_video:
                per_video[video_id] = meta

        for meta in sorted(per_video.values(), key=lambda m: m.get("fetched_at", "")):
            self._register_video(meta)

    # -- public interface (what agents call) ----------------------------------------

    @traceable(run_type="chain", name="ingest_videos")
    def ingest_video(self, url, max_workers: int = 5):
        """Ingest one URL or many. Pass a single URL string to ingest one video
        (returns one status dict), or a list of URLs to ingest them concurrently
        on a bounded thread pool (returns one status dict per URL, in input order).
        """
        if isinstance(url, str):
            return self._ingest_one(url)

        urls = list(url)
        if not urls:
            return []

        results: list = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as ex:
            futures = {ex.submit(self._ingest_one, u): i for i, u in enumerate(urls)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    # _ingest_one already returns dicts for known failures; this
                    # only catches truly unexpected errors so one URL can't sink
                    # the whole batch.
                    results[i] = {
                        "status": "error",
                        "url": urls[i],
                        "reason": f"{type(e).__name__}: {e}",
                    }
        return results
    
    def _ingest_one(self, url: str) -> dict:
        """Full pipeline for a single URL: fetch metadata + transcript, chunk,
        embed, store.

        Returns a status dict in every case — including failures — so the
        supervisor can react instead of the run crashing.
        """
        try:
            video_id = self._get_video_id(url)
        except ValueError as e:
            return {
                "status": "invalid_url",
                "url": url,
                "reason": str(e),
                "action": "Ask the user for a valid, public YouTube URL (watch, youtu.be, or /shorts/).",
            }

        try:
            if not self._is_stale(video_id):
                return {"status": "skipped", "reason": "already up to date", "video_id": video_id}

            try:
                metadata = self._get_metadata(video_id)
            except HttpError as e:
                return {**_youtube_error(e), "video_id": video_id}

            if metadata is None:
                return {
                    "status": "not_found",
                    "video_id": video_id,
                    "reason": "No video could be retrieved for this URL. The video is likely invalid, private, deleted, or region-blocked.",
                    "action": "Do not retry the same URL. If the user named a specific video or channel, search for the closest matching video and confirm it with them before ingesting. Otherwise, ask the user to provide a valid, public YouTube URL.",
                }

            try:
                transcript_data = self._get_transcript(video_id)
            except TranscriptUnavailable as e:
                return {
                    "status": "transcript_unavailable",
                    "video_id": video_id,
                    "reason": str(e),
                    "transcript_available": False,
                    "whisper_status": "unavailable",
                    "metadata_overview": {
                        "title": metadata.get("title"),
                        "channel": metadata.get("channel"),
                        "description": metadata.get("description"),
                        "channel_description": metadata.get("channel_description"),
                        "published_at": metadata.get("published_at"),
                        "view_count": metadata.get("view_count"),
                        "like_count": metadata.get("like_count"),
                        "duration": metadata.get("duration"),
                        "top_comments": self._get_top_comments(video_id),
                    },
                    "action": (
                        "This video could not be ingested: it has no captions/transcript and "
                        "automatic speech-to-text (Whisper) is currently unavailable. Do NOT retry "
                        "ingestion. Be transparent with the user that there is no transcript or "
                        "captions for this video. Then use the fields in metadata_overview (title, "
                        "description, channel_description, and top_comments) to write a brief overview "
                        "of what the video is likely about. Make clear this is inferred from the "
                        "video's metadata and audience comments, not from its actual spoken content."
                    ),
                }

            full_record = {
                **metadata,
                **transcript_data,
                "fetched_at": datetime.now().isoformat()
            }

            chunks = self._chunk_transcript(full_record["transcript"] or "")
            if not chunks:
                return {
                    "status": "empty_transcript",
                    "video_id": video_id,
                    "reason": "The transcript was empty after processing.",
                    "action": "Tell the user there was no spoken content to index for this video.",
                }
            chunk_metadatas = [{k: v for k, v in full_record.items() if k != "transcript"}
                               for _ in chunks]

            # remove any prior chunks for this video so re-ingestion can't duplicate,
            # then add with deterministic IDs so the same chunk always upserts in place.
            # locked so concurrent workers can't corrupt the collection or race on
            # placement assignment in _register_video.
            ids = [f"{video_id}-{i}" for i in range(len(chunks))]
            with self._write_lock:
                self.vectorstore.delete(where={"video_id": video_id})
                self.vectorstore.add_texts(texts=chunks, metadatas=chunk_metadatas, ids=ids)
                entry = self._register_video(full_record)

            return {
                "status": "ingested",
                "video_id": video_id,
                "chunks": len(chunks),
                "lst_placement": entry.lst_placement,
            }
        except Exception as e:
            print(f"[ingest] failed for {video_id}: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "video_id": video_id,
                "reason": f"{type(e).__name__}: {e}",
                "action": "An unexpected error occurred while ingesting. Do not retry the same URL repeatedly; tell the user ingestion failed.",
            }

    def get_video_info(self, placement: int = None, video_id: str = None) -> dict:
        """Quick metadata lookup for an agent tool — no vector search.

        Identify the video by its 1-based list placement (1, 2, ...) or its
        video_id. Returns the flat metadata dict, or a not_found status.
        """
        entry = None
        if placement is not None:
            entry = self.videos.get(placement)
        elif video_id is not None:
            entry = self.videos.get(self._placement_by_id.get(video_id))
        if entry is None:
            return {"status": "not_found", "placement": placement, "video_id": video_id}
        return entry.as_dict()

    def list_videos(self) -> list[dict]:
        """All ingested videos as flat metadata dicts, ordered by placement."""
        return [self.videos[p].as_dict() for p in sorted(self.videos)]

    def search_transcripts(self, query: str, video_id: str = None, k: int = 5) -> list[dict]:
        """Semantic search over ingested transcripts — the supervisor's context source.

        Returns the top-k matching chunks (optionally scoped to one video) as
        dicts of {text, video_id, title, creator, url, placement}, so an agent
        gets both the passage and where it came from for citation/grading.
        """
        try:
            retriever = self.get_retriever(video_id=video_id, k=k)
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"[search_transcripts] query failed: {type(e).__name__}: {e}")
            return []
        results = []
        for doc in docs:
            meta = doc.metadata or {}
            vid = meta.get("video_id")
            results.append({
                "text": doc.page_content,
                "video_id": vid,
                "title": meta.get("title"),
                "creator": meta.get("channel"),
                "url": meta.get("url", f"https://www.youtube.com/watch?v={vid}" if vid else None),
                "placement": self._placement_by_id.get(vid),
            })
        return results

    def search_transcripts_multi(self, searches, k: int = 5, max_workers: int = 5) -> list[dict]:
        """Run several transcript searches at once, concurrently.

        Multi-query counterpart to search_transcripts (same relationship as
        ingest_videos to ingest_video). Each item in `searches` is one search:
        a dict with a "query" (required), an optional "video_id" to scope that
        search to a single video, and an optional "k" to override the default
        chunk count for that search. A bare string is also accepted and treated
        as {"query": <string>}.

        Runs the searches on a bounded thread pool and returns one result set
        per input search, in input order, each a dict of
        {query, video_id, results} where `results` is the list of chunk dicts
        search_transcripts produces. A failure on one search is isolated (its
        results come back empty) so it can't sink the rest of the batch.
        """
        items = list(searches)
        if not items:
            return []

        def _normalize(item: object) -> dict:
            if isinstance(item, str):
                return {"query": item, "video_id": None, "k": k}
            return {
                "query": item.get("query"),
                "video_id": item.get("video_id"),
                "k": item.get("k", k),
            }

        specs = [_normalize(it) for it in items]

        results: list = [None] * len(specs)
        with ThreadPoolExecutor(max_workers=min(max_workers, len(specs))) as ex:
            futures = {
                ex.submit(self.search_transcripts, spec["query"], spec["video_id"], spec["k"]): i
                for i, spec in enumerate(specs)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                spec = specs[i]
                try:
                    hits = fut.result()
                except Exception as e:
                    # search_transcripts already swallows its own errors, so this
                    # only catches the truly unexpected and keeps one bad query
                    # from sinking the batch.
                    print(f"[search_transcripts_multi] query {spec['query']!r} failed: {type(e).__name__}: {e}")
                    hits = []
                results[i] = {
                    "query": spec["query"],
                    "video_id": spec["video_id"],
                    "results": hits,
                }
        return results

    def search_youtube(self, query: str, channel: str = None, max_results: int = 5) -> list[dict]:
        """Search YouTube itself for videos matching a free-text query.

        Use this to find a video that hasn't been ingested yet — it hits the
        YouTube Data API's search endpoint, not the local store. Optionally bias
        toward a creator by passing `channel` (appended to the query). Returns
        candidate {title, creator, video_id, url, published_at} dicts the agent
        can confirm with the user before calling ingest_video.
        """
        q = f"{query} {channel}" if channel else query
        try:
            response = self.youtube.search().list(
                part="snippet",
                q=q,
                type="video",
                maxResults=max_results,
            ).execute()
        except HttpError as e:
            return [_youtube_error(e)]
        results = []
        for item in response.get("items") or []:
            video_id = (item.get("id") or {}).get("videoId")
            if not video_id:
                continue  # channel/playlist results have no videoId
            snippet = item.get("snippet", {})
            results.append({
                "video_id": video_id,
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "title": snippet.get("title"),
                "creator": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
            })
        return results

    def get_tools(self) -> list:
        """LangChain tools bound to this pipeline — the full toolset the supervisor gets.

        Each tool is a closure over `self`, so its schema shows only the real
        arguments, not `self`. Covers the whole pipeline surface the supervisor
        needs: ingest new videos, retrieve context to answer from, find videos on
        YouTube that aren't ingested yet, look up/list metadata, and remove videos.
        Append more here as they're tested.
        """
        @tool
        def ingest_video(url: str) -> dict:
            """Ingest a YouTube video so it can be searched and answered about.

            Fetches metadata + transcript, chunks, embeds, and stores it. Pass a
            full YouTube URL (watch, youtu.be, or /shorts/). Returns a status dict
            with the video_id, chunk count, and list placement; re-ingesting an
            up-to-date video is skipped automatically.
            """
            return self.ingest_video(url)

        @tool
        def ingest_videos(urls: list[str]) -> list[dict]:
            """Ingest several YouTube videos at once (concurrently).

            Use this instead of calling ingest_video repeatedly when the user gives
            more than one URL. Pass a list of full YouTube URLs (watch, youtu.be, or
            /shorts/); returns one status dict per URL, in the same order.
            """
            return self.ingest_video(urls)

        @tool
        def search_transcripts(query: str, video_id: str = None, k: int = 5) -> list[dict]:
            """Semantic search over ingested video transcripts — use this to get
            context before answering a content question.

            Returns the top-k matching transcript chunks, each with its text and
            source (video_id, title, creator, url, placement). Pass a video_id to
            scope the search to a single video, or omit it to search all videos.
            """
            return self.search_transcripts(query=query, video_id=video_id, k=k)

        @tool
        def search_transcripts_multi(searches: list[dict]) -> list[dict]:
            """Run several transcript searches at once (concurrently).

            Use this instead of calling search_transcripts repeatedly when you
            have more than one thing to look up. Pass a list of search objects,
            each shaped like:
                {"query": "<text to search for>", "video_id": "<optional id to scope to one video>"}
            video_id is optional — omit it to search across all videos. Returns
            one result set per search, in the same order, each shaped as
            {query, video_id, results}, where `results` is the usual list of
            matching transcript chunks (text + source).
            """
            return self.search_transcripts_multi(searches)

        @tool
        def search_youtube(query: str, channel: str = None, max_results: int = 5) -> list[dict]:
            """Search YouTube for videos that haven't been ingested yet.

            Use this when the user names a video/topic with no URL and
            search_transcripts finds nothing locally. Hits YouTube directly and
            returns candidate {title, creator, video_id, url, published_at} dicts.
            Optionally pass `channel` to bias toward a creator. Confirm the right
            match with the user, then pass its url to ingest_video.
            """
            return self.search_youtube(query=query, channel=channel, max_results=max_results)

        @tool
        def get_video_info(placement: int = None, video_id: str = None) -> dict:
            """Look up an ingested video's metadata — no vector search.

            Identify the video by its 1-based list placement (1, 2, ...) or its
            video_id. Returns a flat dict (creator, creator_description, title,
            url, view/like counts, duration, ...) or a not_found status.
            """
            return self.get_video_info(placement=placement, video_id=video_id)

        @tool
        def list_videos() -> list[dict]:
            """List every ingested video as a flat metadata dict, ordered by placement."""
            return self.list_videos()

        @tool
        def delete_video(video_id: str) -> dict:
            """Remove an ingested video and all its chunks from the store by video_id.

            Surviving videos keep their list placement. Returns a status dict.
            """
            return self.delete_video(video_id)

        return [ingest_video, ingest_videos, search_transcripts, search_transcripts_multi, search_youtube, get_video_info, list_videos, delete_video]

    def get_retriever(self, video_id: str = None, k: int = 5):
        """Returns a retriever, optionally scoped to a single video."""
        search_kwargs = {"k": k}
        if video_id:
            search_kwargs["filter"] = {"video_id": video_id}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def delete_video(self, video_id: str):
        """Remove all chunks for a video. useful for the management agent."""
        if not video_id:
            return {"status": "error", "reason": "No video_id provided."}
        try:
            self.vectorstore.delete(where={"video_id": video_id})
        except Exception as e:
            print(f"[delete] failed for {video_id}: {type(e).__name__}: {e}")
            return {"status": "error", "video_id": video_id, "reason": f"{type(e).__name__}: {e}"}
        # drop it from the registry too; surviving videos keep their placement
        placement = self._placement_by_id.pop(video_id, None)
        if placement is not None:
            self.videos.pop(placement, None)
        return {"status": "deleted", "video_id": video_id}

    def clear_all(self) -> dict:
        """Wipe every ingested video — drops all vectors and resets the registry.

        Useful for a clean test run. `reset_collection` deletes and recreates the
        underlying Chroma collection, so placements start from 1 again afterwards.
        """
        video_count = len(self.videos)
        try:
            self.vectorstore.reset_collection()
        except Exception as e:
            print(f"[clear_all] failed: {type(e).__name__}: {e}")
            return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        self.videos = {}
        self._placement_by_id = {}
        self._channel_desc_cache = {}
        return {"status": "cleared", "videos_removed": video_count}
    


if __name__ == "__main__":
    pass
