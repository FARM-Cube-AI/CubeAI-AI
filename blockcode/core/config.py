from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
WORKSPACE_DIR = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"

class Settings:
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    
    # 환경 변수에서 CORS 설정 로드 (Docker 환경 대응)
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", 
        '["http://localhost:5173","https://4th-security-cube-ai-fe.vercel.app","http://localhost:5174","http://localhost:9022","http://localhost:9000","http://cubeai.kro.kr/"]'
    )
    
    # 문자열로 받은 경우 JSON 파싱
    if isinstance(CORS_ORIGINS, str):
        import json
        try:
            CORS_ORIGINS = json.loads(CORS_ORIGINS)
        except json.JSONDecodeError:
            CORS_ORIGINS = ["http://localhost:5173"]  # 기본값
    
    # Docker 환경에서 환경 변수로 설정 가능
    HOST = os.getenv("HOST", "0.0.0.0")  # Docker에서는 0.0.0.0으로 바인딩
    PORT = int(os.getenv("PORT", "8000"))  # Docker 컨테이너 내부 포트
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    @staticmethod
    def initialize_directories():
        for directory in [DATASET_DIR, WORKSPACE_DIR, LOGS_DIR]:
            directory.mkdir(exist_ok=True)

settings = Settings()