# ==========================================
# CubeAI 블록코딩 플랫폼 Dockerfile
# ==========================================

FROM python:3.10-slim

# 메타데이터
LABEL maintainer="CubeAI Team"
LABEL description="AI Block Coding Platform - FastAPI Server"
LABEL version="2.0.0"

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필수 도구 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p /app/workspace \
    && mkdir -p /app/dataset \
    && mkdir -p /app/logs

# 권한 설정
RUN chmod -R 755 /app

# 비루트 사용자 생성 및 전환 (보안 강화)
RUN groupadd -r cubeai && useradd -r -g cubeai cubeai
RUN chown -R cubeai:cubeai /app
USER cubeai

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=false

# 헬스체크 설정 (환경 변수 PORT 사용)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://127.0.0.1:${PORT}/ || exit 1

# 애플리케이션 실행
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
