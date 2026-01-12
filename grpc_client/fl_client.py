"""Federated Learning gRPC client."""
import grpc
import io
import torch
import logging
from typing import Dict, Optional
import requests

# Import generated proto files (will be generated later)
try:
    from . import serverside_pb2
    from . import serverside_pb2_grpc
except ImportError:
    # Fallback if proto files not generated yet
    serverside_pb2 = None
    serverside_pb2_grpc = None

logger = logging.getLogger(__name__)


class FederatedLearningClient:
    """Client for Federated Learning server communication."""
    
    def __init__(self, server_url: str, client_id: str):
        """
        Initialize FL client.
        
        Args:
            server_url: gRPC server URL (e.g., "localhost:50051")
            client_id: Unique client identifier
        """
        self.server_url = server_url
        self.client_id = client_id
        self.channel = None
        self.stub = None
    
    def connect(self) -> None:
        """Establish connection to gRPC server."""
        if serverside_pb2_grpc is None:
            raise RuntimeError("Proto files not generated. Run: python -m grpc_tools.protoc ...")
        
        self.channel = grpc.insecure_channel(self.server_url)
        self.stub = serverside_pb2_grpc.AvgWeightsStub(self.channel)
        logger.info(f"Connected to gRPC server at {self.server_url}")
    
    def disconnect(self) -> None:
        """Close connection to gRPC server."""
        if self.channel:
            self.channel.close()
            logger.info("Disconnected from gRPC server")
    
    def upload_weights(self, delta: Dict[str, torch.Tensor], num_examples: int) -> bool:
        """
        Upload local model delta to server.
        
        Args:
            delta: Model weight delta (local - global)
            num_examples: Number of training examples
        
        Returns:
            True if successful, False otherwise
        """
        if self.stub is None:
            logger.error("Not connected to server. Call connect() first.")
            return False
        
        try:
            # Serialize delta to bytes
            buffer = io.BytesIO()
            torch.save(delta, buffer)
            weights_bytes = buffer.getvalue()
            
            # Create gRPC request
            request = serverside_pb2.AddMyWeightsRequest(
                client_id=self.client_id,
                weights=weights_bytes,
                num_examples=num_examples
            )
            
            # Send request
            response = self.stub.AddMyWeights(request)
            logger.info(f"Successfully uploaded weights ({len(weights_bytes)} bytes, {num_examples} examples)")
            return True
        
        except grpc.RpcError as e:
            logger.error(f"gRPC error uploading weights: {e}")
            return False
        except Exception as e:
            logger.error(f"Error uploading weights: {e}")
            return False
    
    def download_global_weights(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Download global model weights from server.
        
        Returns:
            Global model weights or None if failed
        """
        if self.stub is None:
            logger.error("Not connected to server. Call connect() first.")
            return None
        
        try:
            # Request global weights
            request = serverside_pb2.GetReleaseWeightsRequest()
            response = self.stub.GetReleaseWeights(request)
            
            # Get MinIO link
            minio_link = response.link_to_minio
            if not minio_link:
                logger.warning("No global weights available yet")
                return None
            
            logger.info(f"Downloading global weights from: {minio_link}")
            
            # Download from MinIO
            http_response = requests.get(minio_link, timeout=30)
            http_response.raise_for_status()
            
            # Deserialize weights
            buffer = io.BytesIO(http_response.content)
            weights = torch.load(buffer, map_location='cpu')
            
            logger.info(f"Successfully downloaded global weights ({len(http_response.content)} bytes)")
            return weights
        
        except grpc.RpcError as e:
            logger.error(f"gRPC error downloading weights: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"HTTP error downloading from MinIO: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading weights: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
