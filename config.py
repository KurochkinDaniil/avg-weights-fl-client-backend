"""Configuration management using Pydantic Settings."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    
    # Client Configuration
    client_id: str = "client-001"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Server Configuration
    server_grpc_url: str = "localhost:50051"
    
    # Model Configuration
    max_seq_len: int = 300
    input_size: int = 7
    hidden_size: int = 512
    alphabet_size: int = 40
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 5
    
    # Data Storage
    data_dir: Path = Path("./data")
    
    # Alphabet (Russian keyboard + blank)
    # Multi-character tokens are separated by | for parsing
    alphabet: str = "_|й|ц|у|к|е|н|г|ш|щ|з|х|ф|ы|в|а|п|р|о|л|д|ж|э|shift|я|ч|с|м|и|т|ь|б|ю|backspace|toNumberState|globe|,|space|.|enter"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
