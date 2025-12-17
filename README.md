# CubeAI - AI 레포지토리

## 주요 기능

### AI 블록코딩 플랫폼
- **드래그 앤 드롭** 방식으로 AI 모델 설계
- **4단계 학습 커리큘럼**: 새싹 → 잎새 → 가지 → 열매
- **실시간 코드 생성**: 블록 설정을 Python 코드로 자동 변환
- **즉석 실행**: 생성된 코드를 바로 실행하고 결과 확인

### RAG 기반 학습 챗봇
- **AI 학습 전문 어시스턴트**: 단계별 맞춤 가이드 제공
- **문서 기반 답변**: 학습 자료를 바탕으로 정확한 정보 제공
- **모호한 질문 감지**: 애매한 질문 시 구체적인 선택지 제안
- **대화 기록 관리**: 학습 진도에 따른 연속적인 대화 지원


## 빠른 시작

### 1. 환경 설정

`.env` 파일을 생성하고 필요한 환경 변수를 설정하세요:

```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here

# 기타 설정 (선택사항)
DEBUG=false
CORS_ORIGINS=["http://localhost:3000"]
```

### 2. Docker Compose로 실행

```bash
# 전체 서비스 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

### 3. 서비스 접속

**운영 환경 (보안 강화)**
- **블록코딩 플랫폼**: http://localhost:8080
- **챗봇 API**: http://localhost:5000
- **PostgreSQL**: 내부 전용 (외부 접근 차단)
- **Redis**: 내부 전용 (외부 접근 차단)

**개발 환경 (외부 접근 허용)**
```bash
# 개발용 실행 (DB 외부 접근 가능)
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```
- **PostgreSQL**: localhost:5433
- **Redis**: localhost:6379
- **OpenAI API**: http://localhost:8000 (디버깅용)

## 서비스 구성

| 서비스 | 외부 포트 | 설명 |
|--------|----------|------|-----------|
| `blockcode` | 8080 | AI 블록코딩 플랫폼 메인 서버 |
| `conversation` | 5000 | 챗봇 대화 관리 API |
| `openai` | - | RAG 기반 AI 응답 생성 |
| `postgres` | - | pgvector 확장 PostgreSQL |
| `redis` | - | 세션 및 대화 기록 저장소 |

## 🛠️ 개발 환경

### 기술 스택

**Backend**
- FastAPI (Python 3.10)
- LangChain (RAG 구현)
- PyTorch (AI 모델)
- PostgreSQL + pgvector
- Redis

**Infrastructure**
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- AWS ECS (배포)

### 로컬 개발

```bash
# 개별 서비스 개발
cd blockcode
pip install -r requirements.txt
python main.py

# 챗봇만 실행
cd chatbot
docker-compose up -d
```

### Docker Hub 이미지 사용

```bash
# 사전 빌드된 이미지 사용 (DOCKER_USERNAME을 실제 사용자명으로 변경)
docker pull ${DOCKER_USERNAME}/cubeai_blockcode:latest
docker pull ${DOCKER_USERNAME}/cubeai_tutor:latest
docker pull ${DOCKER_USERNAME}/cubeai_openai_rag:latest
docker pull ${DOCKER_USERNAME}/cubeai_postgre:latest

# 또는 docker-compose에서 자동으로 최신 이미지 사용
docker-compose pull
docker-compose up -d
```

## CI/CD 파이프라인

### 자동 배포

- **트리거**: `main` 브랜치 푸시 시 자동 실행
- **Docker Hub**: 이미지 자동 빌드 및 푸시
- **AWS ECS**: 운영 환경 자동 배포
- **보안 스캔**: Trivy를 통한 취약점 검사

### GitHub Secrets 설정

Repository Settings > Secrets에서 다음 값들을 설정하세요:

```
DOCKER_USERNAME      # Docker Hub 사용자명
DOCKER_PASSWORD      # Docker Hub 패스워드
AWS_ACCESS_KEY_ID    # AWS 액세스 키
AWS_SECRET_ACCESS_KEY # AWS 시크릿 키
OPENAI_API_KEY       # OpenAI API 키
```

## 사용법

### 블록코딩 플랫폼

1. **데이터 선택**: CSV 데이터셋 업로드 또는 샘플 데이터 선택
2. **전처리**: 데이터 정규화, 증강 등 전처리 블록 설정
3. **모델 설계**: CNN 레이어 구성 및 하이퍼파라미터 설정
4. **학습 실행**: 모델 훈련 및 실시간 로그 확인
5. **결과 분석**: 정확도, 혼동행렬, 예측 결과 시각화

### 챗봇 사용

```bash
# 새 세션 생성
curl -X POST http://localhost:5000/session

# 대화 전송
curl -X POST http://localhost:5000/conversation/1 \
  -H "Content-Type: application/json" \
  -d '{"conversation": [{"role": "user", "content": "CNN 모델을 어떻게 만드나요?"}]}'
```
