import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

import uvicorn
from config import settings
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run FastAPI server."""
    logger.info(f"Starting FL Client API on {settings.api_host}:{settings.api_port}")
    logger.info(f"Client ID: {settings.client_id}")
    logger.info(f"Data directory: {settings.data_dir}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )


if __name__ == "__main__":
    main()
