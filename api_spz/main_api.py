import logging
import os
import sys
import platform
import torch
from contextlib import asynccontextmanager
# Add the parent directory to sys.path to allow imports from api_spz
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# -------------LOW VRAM TESTING -------------
#
# # only used for debugging, to emulate low-vram graphics cards:
#
# torch.cuda.set_per_process_memory_fraction(0.5)  # Limit to 43% of my available VRAM, for testing.
# And/or set maximum split size (in MB)
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'


import argparse
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Import the StableProjectorz API routes
from api_spz.routes.generation import router as generation_router
from api_spz.core.state_manage import state

# Set up logging
os.makedirs("temp", exist_ok=True)
os.makedirs("temp/current_generation", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("temp", "api.log"))
    ]
)
logger = logging.getLogger("stable3dgen_api")

# Print system information
print(
    f"\n[System Info] Python: {platform.python_version():<8} | "
    f"PyTorch: {torch.__version__:<8} | "
    f"CUDA: {'not available' if not torch.cuda.is_available() else torch.version.cuda}\n"
)

parser = argparse.ArgumentParser(description="Run Stable3DGen (Hi3DGen) - StableProjectorz API server")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
parser.add_argument("--port", type=int, default=7960, help="Port to bind the server to")
parser.add_argument("--model_path", type=str, default="Stable-X/trellis-normal-v0-1", help="Hi3DGen pipeline model path (e.g., local 'weights/trellis-normal-v0-1' or HF ID 'Stable-X/trellis-normal-v0-1')")
parser.add_argument("--normal_predictor_repo", type=str, default="hugoycj/StableNormal", help="torch.hub repo for normal predictor (e.g., 'hugoycj/StableNormal' or a local path to its checkout).")
parser.add_argument("--normal_predictor_model", type=str, default="StableNormal_turbo", help="Model name for normal predictor (e.g., 'StableNormal_turbo').")
parser.add_argument("--normal_predictor_yoso", type=str, default="yoso-normal-v1-8-1", help="YOSO version for normal predictor (e.g., 'yoso-normal-v1-8-1').")
parser.add_argument("--weights_cache_dir", type=str, default="./weights", help="Directory for caching/locating downloaded model weights (used by torch.hub and potentially Hi3DGenPipeline).")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu'). Auto-detects if None.")

args = parser.parse_args()

# Print startup information
print("\n" + "="*50)
print("Stable3DGen (Hi3DGen) - StableProjectorz API Server is starting up:")
print("If it's the first time, neural nets will download. Next runs will be faster.")
print("Touching this window will pause it. If it happens, click inside it and press 'Enter' to unpause")
print("="*50 + "\n")


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # Startup: Initialize models and resources
    final_device = args.device
    if final_device is None:
        final_device = "cuda" if torch.cuda.is_available() else "cpu"
   
    state.initialize_pipeline(
        model_path=args.model_path,
        normal_predictor_repo=args.normal_predictor_repo,
        normal_predictor_model=args.normal_predictor_model,
        yoso_version=args.normal_predictor_yoso,
        weights_cache_dir=args.weights_cache_dir,
        device=final_device
    )
    logger.info(f"Initialized Hi3DGen pipeline from '{args.model_path}' and normal predictor '{args.normal_predictor_repo}/{args.normal_predictor_model}' on {final_device}")
    
    print("\n" + "="*50)
    print(f"Stable3DGen (Hi3DGen) - StableProjectorz API Server v1.0.0")
    print(f"Server is active and listening on {args.host}:{args.port}")
    logger.info(f"Now in StableProjectorz, enter the 3D mode, click on the connection button and enter {args.host}:{args.port}")
    print("="*50 + "\n")
    
    yield  # This is where the FastAPI application runs
    
    # Shutdown: Clean up resources
    state.cleanup()
    logger.info("Server shutting down, resources cleaned up")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Stable3DGen (Hi3DGen) - StableProjectorz API",
    description="API for Hi3DGen 3D generation, compatible with StableProjectorz", 
    version="1.0.0", 
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the generation routes
app.include_router(generation_router)

# Mount static files for downloads
app.mount("/downloads", StaticFiles(directory=Path("temp/current_generation")), name="downloads")


@app.get("/")
async def root():
    return {
        "message": "Stable3DGen (Hi3DGen) - StableProjectorz API is running",
        "status": "ready",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
