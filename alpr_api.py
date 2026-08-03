"""
alpr_api.py

The API layer. Handles receiving a video file over HTTP, saving it
temporarily, calling your processing logic, and returning results.

Run with:
    uvicorn alpr_api:app --reload

NOTE: this will be SLOW to start up, because importing video_processor
loads both YOLO models immediately. That's expected - it only happens
once, when the server starts, not on every request.
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from video_processor import process_video

app = FastAPI(title="ALPR Video API")

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov")


@app.post("/process-video")
async def process_video_endpoint(video: UploadFile = File(...)):
    """
    Upload a video, run the full ALPR pipeline on it, get back
    the detected plates plus a link to download the full CSV.
    """
    if not video.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Please upload a video file: {ALLOWED_EXTENSIONS}"
        )

    # Save the uploaded video to disk with a unique name so
    # simultaneous uploads never overwrite each other
    job_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{video.filename}")

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Run your actual ALPR pipeline - this is the slow part
    csv_path = os.path.join(RESULTS_DIR, f"{job_id}.csv")
    results = process_video(video_path, output_csv_path=csv_path)

    # Summarize results for the API response
    # (the full detail, frame-by-frame, is in the CSV)
    plates_found = set()
    for frame_data in results.values():
        for car_data in frame_data.values():
            plates_found.add(car_data['license_plate']['text'])

    return {
        "job_id": job_id,
        "frames_processed": len(results),
        "unique_plates_detected": len(plates_found),
        "plates": sorted(plates_found),
        "csv_download_url": f"/results/{job_id}"
    }


@app.get("/results/{job_id}")
async def get_results_csv(job_id: str):
    """
    Download the full frame-by-frame CSV for a given job.
    """
    csv_path = os.path.join(RESULTS_DIR, f"{job_id}.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Results not found for this job_id")
    return FileResponse(csv_path, filename=f"alpr_results_{job_id}.csv")


@app.get("/health")
async def health_check():
    return {"status": "ok"}
