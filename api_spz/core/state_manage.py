import logging
import torch
from pathlib import Path
import os
from typing import Optional
from hi3dgen.pipelines import Hi3DGenPipeline

logger = logging.getLogger("stable3dgen_api")


class Hi3DGenState:
    
    def __init__(self):
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.pipeline: Optional[Hi3DGenPipeline] = None
        self.normal_predictor: Optional[torch.nn.Module] = None
        self._device = "cpu"


    def cleanup(self):
        logger.info("Hi3DGenState cleanup: Releasing models.")
        del self.pipeline
        self.pipeline = None
        del self.normal_predictor
        self.normal_predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    def initialize_pipeline(self, 
                           model_path: str,
                           normal_predictor_repo: str, 
                           normal_predictor_model: str = "StableNormal_turbo",
                           yoso_version: str = "yoso-normal-v1-8-1",
                           weights_cache_dir: str = "./weights",
                           device: Optional[str] = None):
        
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Hi3DGen state on device: {self._device}")
        logger.info(f"Pipeline: '{model_path}', NormalPredictor: '{normal_predictor_repo}/{normal_predictor_model}'")
        try:
            # 1. Load Hi3DGenPipeline
            self.pipeline = Hi3DGenPipeline.from_pretrained(model_path)
            self.pipeline.to(self._device).eval()
            logger.info("Hi3DGenPipeline loaded.")

            # 2. Load Normal Predictor (e.g., StableNormal_turbo)
            # This attempts to load from a local path if `normal_predictor_repo` is a valid directory
            # containing hubconf.py, otherwise assumes it's a Hub repository string.
            is_local_repo = Path(normal_predictor_repo).is_dir() and \
                            Path(normal_predictor_repo, "hubconf.py").exists()
            
            hub_source_type = 'local' if is_local_repo else 'github' # 'github' is more generic for hub repos
            repo_to_load = normal_predictor_repo # Path if local, repo string if hub
            
            logger.info(f"Loading Normal Predictor via torch.hub (source: {hub_source_type}, repo: {repo_to_load}). Cache dir: {weights_cache_dir}")
            self.normal_predictor = torch.hub.load(
                repo_or_dir=repo_to_load,
                model=normal_predictor_model,
                source=hub_source_type,
                trust_repo=True, # Essential for custom GitHub repos or local source
                yoso_version=yoso_version,
                local_cache_dir=weights_cache_dir, # For 'github' source, specifies download/cache loc
                pretrained=True
            )
            self.normal_predictor.to(self._device).eval()
            logger.info("Normal Predictor loaded.")

            logger.info("Hi3DGenState initialization successful.")

        except Exception as e:
            logger.error(f"Hi3DGenState initialization failed: {e}", exc_info=True)
            # Ensure partial loads are cleaned up
            del self.pipeline; self.pipeline = None
            del self.normal_predictor; self.normal_predictor = None
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise

# Global state instance
state = Hi3DGenState()