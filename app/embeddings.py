# THIS FILE IS MEANT TO CREATE INITIAL EMBEDDINGS FOR THE REPOSITORIES THAT ARE WATCHED BY THE GIT LINT SERVICE
# IT IS MEANT TO BE RUN ONCE AND THEN THE EMBEDDINGS WILL BE STORED IN PINECONE

import os
import re
import json
import dotenv
from pathlib import Path
from tqdm import tqdm
from hashlib import sha256

from pinecone import Pinecone
from openai import OpenAI

dotenv.load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("git-lint")

openAIClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HASH_CACHE_FILE = "embedded_hashes.json"

def hash_content(text):
    return sha256(text.encode("utf-8")).hexdigest()

def load_hash_cache():
    if Path(HASH_CACHE_FILE).exists():
        return json.loads(Path(HASH_CACHE_FILE).read_text())
    return {}

def save_hash_cache(cache):
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def chunk_code_files(repo_path: str, verbose=True):
    SUPPORTED_EXTENSIONS = {
        ".py": r"(?=def |class )",
        ".js": r"(?=function |class |const |let |var )",
        ".java": r"(?=public |private |protected |class )",
    }

    EXCLUDED_DIRS = {"node_modules", "venv", ".git", "build", "dist", "__pycache__", 
                     "Dockerfile", "docker-compose.yml", "deploy.sh", "requirements.txt",
                     ".env", "data", "README.md", "docs", "logs", "tests", "tmp", "utils", "lib"}

    chunks = []
    skipped_files = 0
    processed_files = 0

    for ext, pattern in SUPPORTED_EXTENSIONS.items():
        for file_path in Path(repo_path).rglob(f"*{ext}"):
            if any(part in EXCLUDED_DIRS for part in file_path.parts):
                if verbose:
                    print(f"❌ Skipping (excluded dir): {file_path}")
                skipped_files += 1
                continue

            try:
                code = file_path.read_text(encoding='utf-8', errors='ignore')
                split_chunks = re.split(pattern, code)

                for i, chunk in enumerate(split_chunks):
                    cleaned = chunk.strip()
                    if len(cleaned) > 50:
                        content_hash = hash_content(cleaned)
                        chunks.append({
                            "id": f"{file_path}-{i}-{content_hash}",
                            "text": cleaned,
                            "metadata": {
                                "path": str(file_path),
                                "chunk_id": i,
                                "hash": content_hash
                            }
                        })

                if verbose:
                    print(f"✅ Processed: {file_path}")
                processed_files += 1

            except Exception as e:
                if verbose:
                    print(f"⚠️ Error reading {file_path}: {e}")
                continue

    if verbose:
        print(f"\nFinished processing.")
        print(f"--Chunks created: {len(chunks)}")
        print(f"--Files processed: {processed_files}")
        print(f"--Files skipped: {skipped_files}")

    return chunks

def embed_chunks(chunks, existing_hashes):
    embedded_chunks = []

    for chunk in tqdm(chunks):
        if chunk["metadata"]["hash"] in existing_hashes:
            print(f"🟢 Skipping existing: {chunk['metadata']['path']} [chunk {chunk['metadata']['chunk_id']}]")
            continue

        response = openAIClient.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        chunk["embedding"] = response.data[0].embedding
        embedded_chunks.append(chunk)
        print(f"🔄 Embedded: {chunk['metadata']['path']} [chunk {chunk['metadata']['chunk_id']}]")

    return embedded_chunks

def upsert_to_pinecone(chunks, index):
    vectors = [
        {
            "id": chunk["id"],
            "values": chunk["embedding"],
            "metadata": chunk["metadata"]
        }
        for chunk in chunks
    ]

    if vectors:
        index.upsert(vectors=vectors)
        print(f"🧠 Upserted {len(vectors)} vectors.")
    else:
        print("🟢 No new vectors to upsert.")

if __name__ == "__main__":
    paths_to_repos = [
        "/Users/kht/repos/StockSense",
        "/Users/kht/repos/portfolio",
        "/Users/kht/repos/MatchPointAI"
    ]

    all_chunks = []
    for path in paths_to_repos:
        chunks = chunk_code_files(path)
        all_chunks.extend(chunks)

    print(f"📊 Total chunks loaded: {len(all_chunks)}")

    existing_hashes = set(load_hash_cache().keys())

    embedded_chunks = embed_chunks(all_chunks, existing_hashes)

    upsert_to_pinecone(embedded_chunks, index)

    # Save newly embedded hashes
    new_hash_cache = {chunk["metadata"]["hash"]: True for chunk in embedded_chunks}
    new_hash_cache.update(load_hash_cache())
    save_hash_cache(new_hash_cache)
