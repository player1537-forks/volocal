#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub>=0.26",
# ]
# ///

"""Build the chunked model repo for Volocal.

Downloads the three source models from their upstream HuggingFace repos,
splits every file into <= 128 MiB chunks, generates ``manifest.json``, and
uploads everything to ``player1537/volocal-models``.

The app then downloads these chunks independently (resumable via HTTP Range)
and reassembles them locally — no more opaque URLSession resume data.

Usage:
    ./scripts/prepare_model_repo.py            # download (if needed), split, upload
    ./scripts/prepare_model_repo.py --no-upload  # just build locally, skip upload
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

DEST_REPO = "player1537/volocal-models"
CHUNK_SIZE = 128 * 1024 * 1024  # 128 MiB

# --- Upstream sources -------------------------------------------------------
# LLM: Unsloth GGUF (single file).
LLM_REPO = "unsloth/Qwen3.5-2B-GGUF"
LLM_FILE = "Qwen3.5-2B-Q4_K_S.gguf"

# STT: Parakeet EOU 320ms variant (3 compiled models + vocab).
STT_REPO = "FluidInference/parakeet-realtime-eou-120m-coreml"
STT_PATTERNS = [
    "320ms/streaming_encoder.mlmodelc/*",
    "320ms/decoder.mlmodelc/*",
    "320ms/joint_decision.mlmodelc/*",
    "320ms/vocab.json",
]

# TTS: PocketTTS (4 compiled models + constants_bin, which holds the tokenizer,
# text embeddings, voice prompts, and Mimi initial state).
TTS_REPO = "FluidInference/pocket-tts-coreml"
TTS_PATTERNS = [
    "cond_step.mlmodelc/*",
    "flowlm_step.mlmodelc/*",
    "flow_decoder.mlmodelc/*",
    "mimi_decoder_v2.mlmodelc/*",
    "constants_bin/*",
]

# Local working directory (git-ignored): source downloads + staging.
WORK = Path(__file__).resolve().parent / ".model-repo"
DOWNLOADS = WORK / "downloads"
STAGING = WORK / "staging"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_snapshot(repo_id: str, patterns: list[str], dest: Path, sentinel: str) -> Path:
    """Download a repo subset unless it's already present locally."""
    marker = dest / sentinel
    if marker.exists():
        print(f"  already present: {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=patterns,
        local_dir=str(dest),
    )
    return dest


def ensure_llm() -> Path:
    """Download the single Unsloth GGUF unless already present locally."""
    dest = DOWNLOADS / "llm"
    gguf = dest / LLM_FILE
    if gguf.exists() and gguf.stat().st_size > 0:
        print(f"  already present: {gguf}")
        return gguf

    dest.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {LLM_REPO}/{LLM_FILE} -> {gguf}")
    hf_hub_download(repo_id=LLM_REPO, filename=LLM_FILE, local_dir=str(dest))
    return gguf


def walk_files(root: Path) -> list[Path]:
    """Recursively collect files, skipping hidden entries (.git, .cache, .DS_Store…)."""
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            out.append(Path(dirpath) / name)
    return sorted(out)


def split_file(src: Path, local_path: str, repo_prefix: str, files: list[dict]) -> None:
    """Split ``src`` into <= CHUNK_SIZE chunks under STAGING and append a manifest entry."""
    size = src.stat().st_size
    whole_hash = hashlib.sha256()
    chunks: list[dict] = []
    idx = 0

    with src.open("rb") as f:
        while True:
            name = f"{repo_prefix}/{local_path}.part{idx:03d}"
            out_path = STAGING / name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            chunk_hash = hashlib.sha256()
            written = 0
            with out_path.open("wb") as out:
                while written < CHUNK_SIZE:
                    block = f.read(min(CHUNK_SIZE - written, 1 << 20))
                    if not block:
                        break
                    out.write(block)
                    chunk_hash.update(block)
                    whole_hash.update(block)
                    written += len(block)

            if written == 0:
                out_path.unlink(missing_ok=True)
                break

            chunks.append(
                {"name": name, "size": written, "sha256": chunk_hash.hexdigest()}
            )
            idx += 1
            if written < CHUNK_SIZE:
                break

    files.append(
        {
            "path": local_path,
            "size": size,
            "sha256": whole_hash.hexdigest(),
            "chunks": chunks,
        }
    )


def build() -> dict:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "chunkSize": CHUNK_SIZE,
        "models": {"llm": [], "stt": [], "tts": []},
    }

    # LLM
    print("LLM:")
    llm = ensure_llm()
    split_file(llm, LLM_FILE, "llm", manifest["models"]["llm"])

    # STT
    print("STT:")
    stt_dir = ensure_snapshot(
        STT_REPO, STT_PATTERNS, DOWNLOADS / "stt", "320ms/vocab.json"
    )
    for path in walk_files(stt_dir):
        rel = path.relative_to(stt_dir).as_posix()  # e.g. 320ms/decoder.mlmodelc/…
        split_file(path, f"parakeet-eou-streaming/{rel}", "stt", manifest["models"]["stt"])

    # TTS
    print("TTS:")
    tts_dir = ensure_snapshot(
        TTS_REPO, TTS_PATTERNS, DOWNLOADS / "tts", "constants_bin/tokenizer.model"
    )
    for path in walk_files(tts_dir):
        rel = path.relative_to(tts_dir).as_posix()  # e.g. cond_step.mlmodelc/…
        split_file(path, f"pocket-tts/{rel}", "tts", manifest["models"]["tts"])

    (STAGING / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for model, files in manifest["models"].items():
        total = sum(f["size"] for f in files)
        chunks = sum(len(f["chunks"]) for f in files)
        print(f"  {model}: {len(files)} file(s), {chunks} chunk(s), {total / 1e9:.2f} GB")

    return manifest


def upload() -> None:
    api = HfApi()
    print(f"Creating/updating repo {DEST_REPO}…")
    api.create_repo(repo_id=DEST_REPO, repo_type="model", exist_ok=True)
    print(f"Uploading {STAGING} -> {DEST_REPO}…")
    api.upload_folder(
        repo_id=DEST_REPO,
        repo_type="model",
        folder_path=str(STAGING),
        commit_message="Update chunked models",
    )
    print("Upload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Build chunks + manifest locally but skip uploading to HuggingFace",
    )
    args = parser.parse_args()

    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    build()
    if args.no_upload:
        print(f"Skipping upload. Artifacts in {STAGING}")
    else:
        upload()


if __name__ == "__main__":
    main()
