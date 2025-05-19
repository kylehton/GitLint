from fastapi import FastAPI, Request, BackgroundTasks
from pinecone import Pinecone
from openai import OpenAI
import os
import dotenv
import httpx
import asyncio
import logging
import re
import json
import boto3
from s3chunks import download_chunk_store_from_s3, load_chunk_store, get_full_chunk_by_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

openAIClient = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
githubKey = os.getenv("GITHUB_ACCESS_TOKEN")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
S3_OBJECT_KEY = "chunk_s3.json"
s3 = boto3.client("s3")

index = pc.Index("git-lint")

systemPrompt = """
    The input is the raw diff of a pull request. You are a meticulous code reviewer with deep expertise in algorithms, 
    data structures, and software engineering best practices.
    IMPORTANT: Please keep all text, analysis and comments, non-verbose, making sure to be concise and to the point, 
    especially for non-impactful changes in the diff.
    Your job:
    Identify every single change, no matter how small (e.g., comment removal, spacing, refactoring).
    Some lines are going to be low impact changes, such as spacing, formatting, comment removal, etc.
    These should NOT be analyzed heavily, and only briefly mentioned at the bottom of the review, before the summary.
    For each impactful changed line, analyze and explain, consolidating analysis where you can, and only mentioning the most impactful changes:
    What was changed.
    Why it was changed (or likely changed).
    Whether the change improves or worsens the code.
    If further improvements or abstractions can be made (e.g., avoid repetition, wasted memory, lack of modularity).
    If no code change is necessary, but improvements are possible (e.g., abstraction opportunities), suggest those.
    Return only the changed lines with explanations — no restating of diffs or unchanged code.
    Do not return code in diff format. Use a human-readable explanation paired directly with the changed lines.
    Your review should help turn the code into the most scalable, efficient, and readable version possible. Assume the 
    author wants direct, precise, and actionable feedback with no fluff. Do not summarize at the start — only provide a 
    detailed final summary at the end of the changes.
    """

app = FastAPI()
background_tasks_set = set()

store_path = download_chunk_store_from_s3()
chunk_store = load_chunk_store(store_path)

def chunk_diff(diff: str, min_len: int = 50):

    chunks = []
    raw_chunks = re.split(r"^diff --git.+?^(@@.+?@@)", diff, flags=re.MULTILINE | re.DOTALL)

    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if len(cleaned) >= min_len:
            chunks.append(cleaned)
    return chunks

async def review_diff(repo_name: str, diff: str):
    try:
        context = await retrieve_context_from_diff(repo_name, diff)
        logger.info(f"Context: {context}")
        prompt = f"{systemPrompt}\n\nContext from codebase:\n{context}\n\nDiff:\n{diff}"
        response = openAIClient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": diff}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error occurred during processing of message: {e}")
        return {"error": str(e)}
    

async def post_comment(issue_url: str, comment: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(issue_url+"/comments", json={"body": comment}, 
            headers={
                "Authorization": f"Bearer {githubKey}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        if response.status_code == 200 or response.status_code == 201:
            return {"message": "Comment posted successfully"}
        else:
            return {"message": "Failed to post comment"}

async def get_diff(url: str):
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        # Code 200 -> Success, 302 -> Redirect
        if response.status_code == 200 or response.status_code == 302:
            return response.text
        else:
            return {"error": "Failed to get pull request diff"}

async def retrieve_context_from_diff(repo_name: str, diff: str, top_k: int = 3):

    chunks = chunk_diff(diff)
    all_matches = []

    for chunk in chunks:
        response = openAIClient.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        vector = response.data[0].embedding

        result = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter={"repo": {"$eq": repo_name}}  # ← filter by repo
        )

        for match in result.get("matches", []):
            chunk_id = match["id"]
            full_chunk = get_full_chunk_by_id(chunk_id, chunk_store)

            if full_chunk:
                logger.info(f"✅ Context match from {match['metadata']['path']} (chunk {match['metadata']['chunk_id']})")
                all_matches.append(full_chunk)
            else:
                logger.warning(f"⚠️ Chunk ID {chunk_id} not found in chunk store")

    return "\n\n".join(all_matches[:5]) 


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    
    # Wait for all background tasks to complete
    if background_tasks_set:
        logger.info(f"Waiting for {len(background_tasks_set)} background tasks to complete")
        print(f"Waiting for {len(background_tasks_set)} background tasks to complete")
        await asyncio.gather(*background_tasks_set, return_exceptions=True)

@app.get("/")
def read_root():
    logger.info("Service is running successfully through EC2 instance of Docker container.")
    return {"Status": "200 OK"}

# Process function for review endpoint
async def process_review(repo_name: str, diff_url: str, issue_url: str):
    try:
        logger.info("[PROCESS]: Retrieving diff from redirect URL")
        diff = await get_diff(diff_url)
        if isinstance(diff, dict) and diff.get("error"):
            logger.error(f"Error getting diff: {diff['error']}")
            return

        logger.info("[PROCESS]: Reviewing diff and creating comment")
        review = await review_diff(repo_name, diff)
        if isinstance(review, dict) and review.get("error"):
            logger.error(f"Error reviewing diff: {review['error']}")
            return

        logger.info("[PROCESS]: Posting comment")
        response = await post_comment(issue_url, review)
        logger.info(f"Comment response: {response}")
    except Exception as e:
        logger.error(f"[ERROR]: {e}")
    
@app.post("/review")
async def webhook(request: Request, background_tasks: BackgroundTasks): 

    logger.info("[/review] Request received")

    if request.headers.get("X-GitHub-Event") == "ping":
        logger.info("[/review] Ping received")
        return {"message": "Ping received!"}
    elif request.headers.get("X-GitHub-Event") == "pull_request":
        data = await request.json()

        full_repo = data["repository"]["full_name"]  
        repo_name = full_repo.split("/")[-1] 
        diff_url = data["pull_request"]["diff_url"]
        issue_url = data["pull_request"]["issue_url"]
        
        background_tasks.add_task(process_review, repo_name, diff_url, issue_url)
        logger.info("[/review] Responding immediately")
        
        return {"message": "Review started, response will be posted shortly."}
    else:
        logger.info("[/review] Unknown event received")
        return {"message": "Unknown event received"}