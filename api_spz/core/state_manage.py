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
            if self.pipeline is None: 
                raise RuntimeError(f"Hi3DGenPipeline.from_pretrained('{model_path}') returned None.")
            
            self.pipeline.to(self._device)  # Moves all sub-models in self.pipeline.models

            # Explicitly set all sub-models within the pipeline to eval mode
            if hasattr(self.pipeline, 'models') and self.pipeline.models:
                for model_name, model_instance in self.pipeline.models.items():
                    if hasattr(model_instance, 'eval') and callable(model_instance.eval):
                        model_instance.eval()
                logger.info("All sub-models in Hi3DGenPipeline set to evaluation mode.")
            else:
                logger.warning("Hi3DGenPipeline.models not found or empty; could not set sub-models to eval mode explicitly.")
            
            logger.info("Hi3DGenPipeline loaded and configured.")

            # 2. Load Normal Predictor (e.g., StableNormal_turbo)
            is_local_repo = Path(normal_predictor_repo).is_dir() and \
                            Path(normal_predictor_repo, "hubconf.py").exists()
            
            hub_source_type = 'local' if is_local_repo else 'github'
            repo_to_load = normal_predictor_repo
            
            logger.info(f"Loading Normal Predictor via torch.hub (source: {hub_source_type}, repo: {repo_to_load}). Cache dir: {weights_cache_dir}")
            self.normal_predictor = torch.hub.load(
                repo_or_dir=repo_to_load,
                model=normal_predictor_model,
                source=hub_source_type,
                trust_repo=True, 
                yoso_version=yoso_version,
                local_cache_dir=weights_cache_dir,
                #pretrained=True  REMOVED THIS LINE based on app.py's fallback and the error
            )
            if self.normal_predictor is None: 
                raise RuntimeError(f"torch.hub.load for '{normal_predictor_model}' from '{repo_to_load}' returned None.")
            self.normal_predictor.to(self._device) 
            
            # Explicitly set all sub-models within the normal_predictor to eval mode
            if hasattr(self.normal_predictor, 'models') and self.normal_predictor.models:
                for model_name, model_instance in self.normal_predictor.models.items():
                    if hasattr(model_instance, 'eval') and callable(model_instance.eval):
                        model_instance.eval()
                logger.info("All sub-models in normal_predictor set to evaluation mode.")

            logger.info("Normal Predictor loaded and configured.")

            logger.info("Hi3DGenState initialization successful.")

        except Exception as e:
            logger.error(f"Hi3DGenState initialization failed: {e}", exc_info=True)
            if hasattr(self, 'pipeline') and self.pipeline is not None: del self.pipeline
            self.pipeline = None
            if hasattr(self, 'normal_predictor') and self.normal_predictor is not None: del self.normal_predictor
            self.normal_predictor = None
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise

# Global state instance
state = Hi3DGenState()