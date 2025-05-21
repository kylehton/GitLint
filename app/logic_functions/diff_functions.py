from logic_functions.s3_upload import download_chunk_store_from_s3, load_chunk_store, get_full_chunk_by_id, save_chunk_store_locally, upload_chunk_store_to_s3
from logic_functions.embeddings import chunk_code_files, embed_chunks, upsert_to_pinecone, hash_content

from pinecone import Pinecone
from openai import OpenAI
import os
import dotenv
import httpx
import logging
import re
import boto3

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

store_path = download_chunk_store_from_s3()
chunk_store = load_chunk_store(store_path)

systemPrompt = """
    The input is the raw diff of a pull request. You are a meticulous code reviewer with deep expertise in algorithms, 
    data structures, and software engineering best practices.
    IMPORTANT: Please keep all text, analysis and comments, non-verbose, making sure to be concise and to the point, 
    especially for non-impactful changes in the diff.
    Your job:
        Identify every single change, no matter how small (e.g., comment removal, spacing, refactoring).
        Some lines are going to be low impact changes, such as spacing, formatting, comment removal, etc.
        These should NOT be analyzed heavily, and only briefly mentioned at the bottom of the review, before the summary.
        Impactful changes are: changes to logic and functionality, adding or removing features, and those types of changes
        that have a significant impact on the codebase and how it functions.
        For each impactful changed line, analyze and explain, consolidating analysis where you can, and only mentioning the most 
        impactful changes:
            - What was changed.
            - Why it was changed (or likely changed).
            - Whether the change improves or worsens the code.
            - If further improvements or abstractions can be made (e.g., avoid repetition, wasted memory, lack of modularity).
            - If no code change is necessary, but improvements are possible (e.g., abstraction opportunities), suggest those.
            - Return only the changed lines with explanations — no restating of diffs or unchanged code.
            - Do not return code in diff format. Use a human-readable explanation paired directly with the changed lines.
        Your review should help turn the code into the most scalable, efficient, and readable version possible. Assume the 
        author wants direct, precise, and actionable feedback with no fluff. Do not summarize at the start — only provide a 
        detailed final summary at the end of the changes.
    """

##### FUNCTIONS (are written in order they are called in function pipeline) #####

### CALLED BY: main.py
### PURPOSE: Handles function chain to process a pull request diff and generate a review comment
# 1. Retrieve the diff from the redirect URL
# 2. Review the diff and create a comment
# 3. Post the comment to the issue
# @param repo_name: str - The name of the repository
# @param diff_url: str - The URL of the diff
# @param issue_url: str - The URL of the issue
# @return: None
async def process_review(repo_name: str, diff_url: str, issue_url: str):
    try:
        logger.info("[PROCESS]: Retrieving diff from redirect URL")
        # 1. Retrieve the diff from the redirect URL
        diff = await get_diff(diff_url)
        if isinstance(diff, dict) and diff.get("error"):
            logger.error(f"Error getting diff: {diff['error']}")
            return

        logger.info("[PROCESS]: Reviewing diff and creating comment")
        # 2. Review the diff and create a comment
        review = await review_diff(repo_name, diff)
        if isinstance(review, dict) and review.get("error"):
            logger.error(f"Error reviewing diff: {review['error']}")
            return

        logger.info("[PROCESS]: Posting comment")
        # 3. Post the comment to the issue
        response = await post_comment(issue_url, review)
        logger.info(f"Comment response: {response}")

        # 4. Update embeddings for modified files
        if response.get("message") == "Comment posted successfully":
            logger.info("[PROCESS]: Updating embeddings for modified files")
            await update_file_embeddings(repo_name, diff)
            logger.info("[PROCESS]: Successfully updated embeddings")
    except Exception as e:
        logger.error(f"[ERROR]: {e}")


### CALLED BY: process_review
### PURPOSE: Retrieves the diff from the redirect URL to be used as the input for the review
# 1. Retrieve the diff from the redirect URL
# 2. Return the diff in text/string format
# @param url: str - The URL of the diff
# @return: str - The diff as text
async def get_diff(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(url)
        # Code 200 -> Success, 302 -> Redirect
        if response.status_code == 200 or response.status_code == 302:
            return response.text
        else:
            return {"error": "Failed to get pull request diff"}


### CALLED BY: process_review
### PURPOSE: Calls gpt-4o-mini to generate a code review comment based on the prompt, diff, and the codebase context
# 1. Retrieve the context from the diff
# 2. Call gpt-4o-mini to generate a code review comment
# 3. Return the code review comment
# @param repo_name: str - The name of the repository to be searched for context
# @param diff: str - The diff of the pull request
# @return: str - code review comment generated by LLM
async def review_diff(repo_name: str, diff: str) -> str:
    try:
        # 1. Retrieve the context from the diff
        context = await retrieve_context_from_diff(repo_name, diff)
        logger.info(f"Context: {context}")

        # 2. Call gpt-4o-mini to generate a code review comment
        prompt = f"{systemPrompt}\n\nContext from codebase:\n{context}\n\nDiff:\n{diff}"
        response = openAIClient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": diff}
            ]
        )
        # 3. Return the code review comment
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error occurred during processing of message: {e}")
        return {"error": str(e)}
    

### CALLED BY: review_diff
### PURPOSE: Retrieves the context from the diff by searching the vector database for the most relevant chunks
# 1. Retrieve the file paths from the diff
# 2. Embed sizable chunks of the diff from chunk_diff()
# 3. Query the vector database for the most relevant chunks to the diff
# 4. Concatenate the most relevant chunks and return them
# @param repo_name: str - The name of the repository to be searched for context
# @param diff: str - The diff of the pull request
# @param top_k: int - The number of chunks to concatenate and return
# @return: str - concatenated string of context from the codebase
async def retrieve_context_from_diff(repo_name: str, diff: str, top_k: int = 3) -> str:
    try:
        # 1. Retrieve the file paths from the diff
        file_paths = extract_file_paths_from_diff(diff)

        # 2. Embed sizable chunks of the diff from chunk_diff()
        chunks = chunk_diff(diff)
        all_matches = []

        # 3. Embed each chunk and query the vector database for the most relevant chunks to the diff
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
                filter={"repo": {"$eq": repo_name},
                        "path": {"$in": list(file_paths)}}  
            )

            # 4. Append the most relevant chunks to the list
            for match in result.get("matches", []):
                chunk_id = match["id"]
                full_chunk = get_full_chunk_by_id(chunk_id, chunk_store)

                if full_chunk:
                    # Log the location of the context match
                    logger.info(f"✅ Context match from {match['metadata']['path']} (chunk {match['metadata']['chunk_id']})")
                    all_matches.append(full_chunk)
                else:
                    logger.warning(f"⚠️ Chunk ID {chunk_id} not found in chunk store")

        # TODO: Retrieve S3 embeddings that directly correspond to where the chunks are making changes to
        # TODO: Create S3 function to re-upload relevant chunks with new changes made to them

        return "\n\n".join(all_matches[:3])
    
    except Exception as e:
        logger.error(f"Error occurred during retrieval of context from diff: {e}")
        return {"error": str(e)}


### CALLED BY: retrieve_context_from_diff
### PURPOSE: Extracts the file paths from the diff to be used as a filter for the context search
# 1. Extract the file paths from the diff
# 2. Append the file paths into a set, and return the set
# @param diff: str - The diff of the pull request
# @return: set[str] - The file paths from the diff
def extract_file_paths_from_diff(diff: str) -> set[str]:
    paths = set()
    for line in diff.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/", line)
        if match:
            paths.add(match.group(1))
    return paths 


### CALLED BY: retrieve_context_from_diff
### PURPOSE: Splits the entire diff into chunks, which, if longer than min_len, are added to the vector database
# 1. Chunk the diff
# 2. Measure the length of each chunk to check usefulness as an embedding
# 3. Append the chunks into a list, and return the list
# @param diff: str - The diff of the pull request
# @param min_len: int - The minimum length of a chunk
# @return: list[str] - The chunks of the diff
def chunk_diff(diff: str, min_len: int = 50) -> list[str]:

    chunks = []
    raw_chunks = re.split(r"^diff --git.+?^(@@.+?@@)", diff, flags=re.MULTILINE | re.DOTALL)

    for chunk in raw_chunks:
        cleaned = chunk.strip()
        if len(cleaned) >= min_len:
            chunks.append(cleaned)
    return chunks


### CALLED BY: process_review
### PURPOSE: Posts the comment to the issue
# 1. Post the comment to the issue
# 2. Return a success message
# @param issue_url: str - The URL of the issue
# @param comment: str - The comment to be posted
# @return: dict - The response from the API
async def post_comment(issue_url: str, comment: str) -> dict:
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

async def get_file_content(repo_name: str, file_path: str) -> str:
    try:
        url = f"https://raw.githubusercontent.com/kylehton/{repo_name}/main/{file_path}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"Failed to get file from {url}: {response.status_code}")
                return None
    except Exception as e:
        logger.error(f"Error getting file content: {e}")
        return None

async def update_file_embeddings(repo_name: str, diff: str):
    global chunk_store
    
    try:
        # Get modified file paths from diff
        file_paths = extract_file_paths_from_diff(diff)
        if not file_paths:
            logger.info("No files to update")
            return

        # Process each modified file
        for file_path in file_paths:
            # Get file content from GitHub
            content = await get_file_content(repo_name, file_path)
            if not content:
                logger.warning(f"Could not get content for {file_path}")
                continue

            # Delete existing chunks for this file
            chunks_to_delete = []
            for chunk_id, chunk_data in chunk_store.items():
                if chunk_data.get("path") == file_path:
                    chunks_to_delete.append(chunk_id)
            
            if chunks_to_delete:
                try:
                    # Delete from Pinecone
                    index.delete(ids=chunks_to_delete)
                    # Remove from store
                    for chunk_id in chunks_to_delete:
                        chunk_store.pop(chunk_id, None)
                    logger.info(f"Deleted {len(chunks_to_delete)} chunks for {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting chunks for {file_path}: {e}")

            # Create chunks from file content
            chunks = []
            # Use the same chunking patterns as in embeddings.py
            patterns = {
                ".py": r"(?=def |class )",
                ".js": r"(?=function |class |const |let |var )",
                ".java": r"(?=public |private |protected |class )",
            }
            
            ext = os.path.splitext(file_path)[1]
            if ext in patterns:
                split_chunks = re.split(patterns[ext], content)
                for i, chunk in enumerate(split_chunks):
                    cleaned = chunk.strip()
                    if len(cleaned) > 50:
                        # Use the same ID format as embeddings.py
                        content_hash = hash_content(cleaned)
                        chunks.append({
                            "id": f"{file_path} (chunk {i}.0)-{content_hash}",
                            "text": cleaned,
                            "metadata": {
                                "path": file_path,
                                "chunk_id": i,
                                "hash": content_hash,
                                "repo": repo_name,
                                "preview": cleaned[:200]
                            }
                        })

            if not chunks:
                logger.warning(f"No chunks created for {file_path}")
                continue

            # Embed the chunks
            embedded_chunks = []
            for chunk in chunks:
                response = openAIClient.embeddings.create(
                    input=chunk["text"],
                    model="text-embedding-3-small"
                )
                chunk["embedding"] = response.data[0].embedding
                embedded_chunks.append(chunk)
                logger.info(f"Embedded: {chunk['metadata']['path']} [chunk {chunk['metadata']['chunk_id']}]")

            # pre-check the chunks before local upsert
            for i, chunk in enumerate(embedded_chunks):
                if not isinstance(chunk, dict):
                    logger.error(f"❌ embedded_chunks[{i}] is not a dict: {chunk}")
                elif "metadata" not in chunk or "text" not in chunk:
                    logger.error(f"❌ embedded_chunks[{i}] missing keys: {chunk.keys()}")

            # upsert the chunks to pinecone
            upsert_to_pinecone(embedded_chunks, index)

            
            # Update local store
            for chunk in embedded_chunks:
                chunk_store[chunk["id"]] = {
                    "text": chunk["text"],
                    "path": chunk["metadata"]["path"],
                    "chunk_id": chunk["metadata"]["chunk_id"]
                }

        # Save updated store
        save_chunk_store_locally(chunk_store)
        upload_chunk_store_to_s3()
        
        logger.info(f"Successfully updated embeddings for {len(file_paths)} files")
        
    except Exception as e:
        logger.error(f"Error updating file embeddings: {e}")
        raise