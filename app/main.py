from diff_functions import process_review
from fastapi import FastAPI, Request, BackgroundTasks
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
background_tasks_set = set()

### FASTAPI ENDPOINTS ###

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")
    # Wait for all background tasks to complete
    if background_tasks_set:
        logger.info(f"Waiting for {len(background_tasks_set)} background tasks to complete")
        await asyncio.gather(*background_tasks_set, return_exceptions=True)

@app.get("/")
def read_root():
    logger.info("Service is running successfully through EC2 instance of Docker container.")
    return {"Status": "200 OK"}
    
@app.post("/review")
async def webhook(request: Request, background_tasks: BackgroundTasks): 
    logger.info("[/review] Request received")

    # Handle ping event from GitHub Webhook
    if request.headers.get("X-GitHub-Event") == "ping":
        logger.info("[/review] Ping received")
        return {"message": "Ping received!"}
    elif request.headers.get("X-GitHub-Event") == "pull_request":
        data = await request.json()
        full_repo = data["repository"]["full_name"]  
        repo_name = full_repo.split("/")[-1] # Parse repo name for custom filter search
        diff_url = data["pull_request"]["diff_url"]
        issue_url = data["pull_request"]["issue_url"]
        
        # Call function chain to process diff and generate a review comment
        background_tasks.add_task(process_review, repo_name, diff_url, issue_url)
        logger.info("[/review] Responding immediately")
        
        # Return response to GitHub to confirm receiving Pull Request webhook
        return {"message": "Review started, response will be posted shortly."}
    else:
        # Return error message if event is not a ping or pull request
        logger.info("[/review] Unknown event received")
        return {"message": "Unknown event received"}