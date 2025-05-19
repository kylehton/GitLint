import os
import json
import boto3
import dotenv

from pathlib import Path

dotenv.load_dotenv()

CHUNK_STORE_FILE = "chunk_s3.json"
S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_OBJECT_KEY = "chunk_s3.json"

s3 = boto3.client("s3")

# ---------------------------
# Save locally & upload to S3
# ---------------------------

def save_chunk_store_locally(chunks):
    store = {
        chunk["id"]: {
            "text": chunk["text"],
            "path": chunk["metadata"]["path"],
            "chunk_id": chunk["metadata"]["chunk_id"]
        }
        for chunk in chunks
    }

    with open(CHUNK_STORE_FILE, "w") as f:
        json.dump(store, f, indent=2)

def upload_chunk_store_to_s3():
    if not Path(CHUNK_STORE_FILE).exists():
        raise FileNotFoundError(f"{CHUNK_STORE_FILE} does not exist")
    
    s3.upload_file(CHUNK_STORE_FILE, S3_BUCKET, S3_OBJECT_KEY)
    print(f"✅ Uploaded chunk store to s3://{S3_BUCKET}/{S3_OBJECT_KEY}")

# ---------------------------
# Download & Load in FastAPI
# ---------------------------

def download_chunk_store_from_s3(dest_path="/tmp/chunk_store.json"):
    s3.download_file(S3_BUCKET, S3_OBJECT_KEY, dest_path)
    print(f"✅ Downloaded chunk store to {dest_path}")
    return dest_path

def load_chunk_store(path="/tmp/chunk_store.json"):
    with open(path) as f:
        return json.load(f)

def get_full_chunk_by_id(chunk_id, store):
    return store.get(chunk_id, {}).get("text", "")
