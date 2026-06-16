'''This module holds the agent that ingests videos using the given URL, vectorises them, and adds them to the Chromadb database'''

import os
from dotenv import load_dotenv
from datetime import datetime

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from requests.exceptions import RequestException
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from urllib.parse import urlparse, parse_qs
from langchain_core.tools import tool
from googleapiclient import discovery
from langchain_chroma import Chroma
import chromadb, whisper, yt_dlp

from langsmith.wrappers import wrap_gemini

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.proxies import WebshareProxyConfig

from func import VideoList


load_dotenv()
# FREE_API = os.getenv("FREE_GEMINI_API")
# PAID_API = os.getenv("PAID_GEMINI_API")
UTUBE_API = os.getenv("YOUTUBE_API_KEY")
WS_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME")
WS_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD")

class VideoIngestionPipeline:
    def __init__(self, google_api_key: str):
        self.max_age_days = 7
        self.youtube = wrap_gemini(discovery.build("youtube", "v3", developerKey=google_api_key))
        self.ytt_api = YouTubeTranscriptApi(proxy_config=WebshareProxyConfig(
                                            proxy_username=WS_USERNAME,
                                            proxy_password=WS_PASSWORD,
                                            ))
        self.whisper_model = whisper.load_model("base")
        self.embeddings = wrap_gemini(GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        ))
        self.vectorstore = Chroma(
            collection_name="youtube_videos",
            embedding_function=self.embeddings,
            # anchored to app/data so the same DB is used no matter
            # which working directory the app is launched from
            persist_directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")
        )

        # quick-access metadata registry (what agent tools read instead of
        # searching the vector store): placement -> VideoList, plus a
        # video_id -> placement index so re-ingestion keeps a stable slot.
        self.videos: dict[int, VideoList] = {}
        self._placement_by_id: dict[str, int] = {}
        # channel descriptions are reused across a creator's videos; cache them
        # so we don't pay an extra YouTube call per video for the same channel.
        self._channel_desc_cache: dict[str, str] = {}
        self._rebuild_registry()

    # -- private helpers ------------------------------------------------------------

    def _get_video_id(self, url: str) -> str:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        if "/shorts/" in url:
            return url.split("/shorts/")[-1].split("?")[0]
        return parse_qs(urlparse(url).query)["v"][0]

    def _get_channel_description(self, channel_id: str) -> str:
        """Channel ('creator') description, cached by channel_id."""
        if channel_id in self._channel_desc_cache:
            return self._channel_desc_cache[channel_id]
        response = self.youtube.channels().list(
            part="snippet",
            id=channel_id
        ).execute()
        items = response.get("items") or []
        description = items[0]["snippet"]["description"] if items else ""
        self._channel_desc_cache[channel_id] = description
        return description

    def _get_metadata(self, video_id: str) -> dict:
        response = self.youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        ).execute()
        item = response["items"][0]
        snippet, stats = item["snippet"], item["statistics"]
        return {
            "video_id":    video_id,
            "url":         f"https://www.youtube.com/watch?v={video_id}",
            "title":       snippet["title"],
            "channel":     snippet["channelTitle"],
            "channel_id":  snippet["channelId"],
            "channel_description": self._get_channel_description(snippet["channelId"]),
            "published_at": snippet["publishedAt"],
            "view_count":  stats.get("viewCount"),
            "like_count":  stats.get("likeCount"),
            "duration":    item["contentDetails"]["duration"],
        }

    def _get_transcript(self, video_id: str) -> dict:
        try:
            fetched = self.ytt_api.fetch(video_id)
            return {
                "transcript": " ".join([s.text for s in fetched.snippets]),
                "transcript_source": "captions"
            }
        except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript, RequestException) as e:
            # No captions, captions disabled, or the request was blocked/rate-limited.
            # RequestException covers RetryError ("too many 429 error responses") raised
            # when the timedtext endpoint rate-limits the proxy IP. Fall back to
            # downloading the audio and transcribing it locally with Whisper, which
            # never touches the rate-limited timedtext endpoint.
            print(f"[transcript] captions unavailable for {video_id} ({type(e).__name__}); falling back to Whisper")
            return {
                "transcript": self._whisper_transcribe(video_id),
                "transcript_source": "whisper"
            }

    def _whisper_transcribe(self, video_id: str) -> str:
        url = f"https://www.youtube.com/watch?v={video_id}"
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": video_id,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        result = self.whisper_model.transcribe(f"{video_id}.mp3")
        os.remove(f"{video_id}.mp3")
        return result["text"]

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

        stored = self.vectorstore.get()
        metadatas = stored.get("metadatas") or []

        # collapse chunk metadata down to one entry per video
        per_video: dict[str, dict] = {}
        for meta in metadatas:
            video_id = meta.get("video_id")
            if video_id and video_id not in per_video:
                per_video[video_id] = meta

        for meta in sorted(per_video.values(), key=lambda m: m.get("fetched_at", "")):
            self._register_video(meta)

    # -- public interface (what agents call) ----------------------------------------

    def ingest_video(self, url: str) -> dict:
        """Full pipeline: fetch metadata + transcript, chunk, embed, store."""
        video_id = self._get_video_id(url)

        if not self._is_stale(video_id):
            return {"status": "skipped", "reason": "already up to date", "video_id": video_id}

        metadata = self._get_metadata(video_id)
        transcript_data = self._get_transcript(video_id)

        full_record = {
            **metadata,
            **transcript_data,
            "fetched_at": datetime.now().isoformat()
        }

        chunks = self._chunk_transcript(full_record["transcript"])
        chunk_metadatas = [{k: v for k, v in full_record.items() if k != "transcript"}
                           for _ in chunks]

        # remove any prior chunks for this video so re-ingestion can't duplicate,
        # then add with deterministic IDs so the same chunk always upserts in place
        self.vectorstore.delete(where={"video_id": video_id})
        ids = [f"{video_id}-{i}" for i in range(len(chunks))]
        self.vectorstore.add_texts(texts=chunks, metadatas=chunk_metadatas, ids=ids)

        entry = self._register_video(full_record)

        return {
            "status": "ingested",
            "video_id": video_id,
            "chunks": len(chunks),
            "lst_placement": entry.lst_placement,
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

    def get_tools(self) -> list:
        """LangChain tools bound to this pipeline — what an agent gets handed.

        Each tool is a closure over `self`, so its schema shows only the real
        arguments (placement / video_id), not `self`. Start with these two;
        append more here as they're tested.
        """
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

        return [get_video_info, list_videos]

    def get_retriever(self, video_id: str = None, k: int = 5):
        """Returns a retriever, optionally scoped to a single video."""
        search_kwargs = {"k": k}
        if video_id:
            search_kwargs["filter"] = {"video_id": video_id}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def delete_video(self, video_id: str):
        """Remove all chunks for a video. useful for the management agent."""
        self.vectorstore.delete(where={"video_id": video_id})
        # drop it from the registry too; surviving videos keep their placement
        placement = self._placement_by_id.pop(video_id, None)
        if placement is not None:
            self.videos.pop(placement, None)
        return {"status": "deleted", "video_id": video_id}
    


if __name__ == "__main__":
    pass
