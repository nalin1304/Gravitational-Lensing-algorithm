# Scalability & Performance Analysis

**Document Version:** 1.0  
**Date:** October 7, 2025  
**Focus:** Load Handling, Bottlenecks, Performance Optimization  
**Test Environment:** Current Docker Compose Setup

---

## Executive Summary

### Current Limitations
- **Single-instance deployment** - No horizontal scaling
- **No load balancing** - Direct container access
- **Synchronous processing** - Blocking operations
- **Limited concurrency** - Resource contention under load
- **No caching strategy** - Redundant computations

### Target Metrics (Production-Ready)
| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Concurrent Users | ~10 | 1,000+ | **100x** |
| Request Latency (p95) | Unknown | <200ms | Need testing |
| Throughput (req/s) | Unknown | 1,000+ | Need testing |
| Uptime | Single point of failure | 99.9% | Need HA |
| Analysis Queue Time | Synchronous | <5 min | Need async |

---

## 1. Current Architecture Bottlenecks

### 1.1 API Layer Bottlenecks

#### Issue: Single Uvicorn Worker
**Location:** `docker-compose.yml` and `Dockerfile`

**Current Configuration:**
```dockerfile
# Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Single worker, synchronous processing
```

**Problem:** 
- Only 1 worker process handling all requests
- CPU-bound operations (PINN inference, ray tracing) block other requests
- Maximum concurrent requests limited by worker's event loop

**Load Testing Results (Estimated):**
```
Scenario: 100 concurrent users making analysis requests

Current Setup (1 worker):
├─ Requests per second: ~10 req/s
├─ Average latency: 2-5 seconds
├─ P95 latency: 10-15 seconds
├─ Error rate: 0% (but slow)
└─ CPU usage: 100% (saturated)

With 4 Workers:
├─ Requests per second: ~40 req/s (4x improvement)
├─ Average latency: 500ms - 1s
├─ P95 latency: 2-3 seconds
├─ Error rate: 0%
└─ CPU usage: 80-90% per core
```

**Remediation:**
```dockerfile
# Dockerfile - Multi-worker setup
# Calculate workers: (2 x CPU cores) + 1
CMD ["gunicorn", "api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
```

```txt
# requirements.txt - Add gunicorn
gunicorn>=21.2.0
```

#### Issue: No Request Timeout Configuration

**Problem:** Long-running analysis requests can hold connections indefinitely.

**Remediation:**
```python
# api/main.py - Add timeout middleware
from fastapi import Request
import asyncio

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=60.0)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"detail": "Request timeout"}
        )
```

#### Issue: No Connection Pooling

**Problem:** Each request creates new database connection.

**Current:**
```python
# api/analysis_routes.py
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Optimization:**
```python
# database/database.py - Add connection pooling
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Connections to keep open
    max_overflow=10,       # Additional connections in high load
    pool_pre_ping=True,    # Verify connection before use
    pool_recycle=3600,     # Recycle connections after 1 hour
    echo=False
)
```

### 1.2 Database Bottlenecks

#### Issue: No Indexing Strategy

**Current Schema (database/models.py):**
```python
class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_type = Column(String)
    created_at = Column(DateTime)
    # No additional indexes!
```

**Problem:** Queries on `user_id`, `analysis_type`, `created_at` will do full table scans.

**Load Impact:**
```
Query: SELECT * FROM analyses WHERE user_id = 123 ORDER BY created_at DESC

Without Indexes:
├─ Execution time: 500ms (for 100K records)
├─ DB CPU: 60%
└─ Locks: Shared lock on entire table

With Indexes:
├─ Execution time: 5ms (100x faster)
├─ DB CPU: 2%
└─ Locks: Row-level locks only
```

**Remediation:**
```python
# database/models.py - Add strategic indexes
class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        Index('idx_user_created', 'user_id', 'created_at'),  # Composite
        Index('idx_analysis_type', 'analysis_type'),
        Index('idx_status', 'status'),
    )
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    analysis_type = Column(String, index=True)
    status = Column(String, index=True)  # For queuing
    created_at = Column(DateTime, index=True)
```

```bash
# Create migration
alembic revision -m "add_performance_indexes"
```

```python
# migrations/versions/xxx_add_performance_indexes.py
def upgrade():
    op.create_index('idx_user_created', 'analyses', ['user_id', 'created_at'])
    op.create_index('idx_analysis_type', 'analyses', ['analysis_type'])
    op.create_index('idx_status', 'analyses', ['status'])

def downgrade():
    op.drop_index('idx_user_created', 'analyses')
    op.drop_index('idx_analysis_type', 'analyses')
    op.drop_index('idx_status', 'analyses')
```

#### Issue: No Query Optimization

**Current:** N+1 query problem
```python
# api/analysis_routes.py
@router.get("/analyses")
async def list_analyses(current_user: User = Depends(get_current_user)):
    analyses = db.query(Analysis).filter(Analysis.user_id == current_user.id).all()
    
    # For each analysis, fetch user (N queries!)
    for analysis in analyses:
        user = db.query(User).filter(User.id == analysis.user_id).first()
```

**Optimized:**
```python
from sqlalchemy.orm import joinedload

@router.get("/analyses")
async def list_analyses(current_user: User = Depends(get_current_user)):
    # Single query with eager loading
    analyses = db.query(Analysis)\
        .filter(Analysis.user_id == current_user.id)\
        .options(joinedload(Analysis.user))\
        .all()
```

#### Issue: No Read Replicas

**Problem:** All read and write queries hit primary database.

**Architecture (Proposed):**
```
┌─────────────┐
│  API Layer  │
└──────┬──────┘
       │
       ├──────────────────┬──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐   ┌─────────────┐
│  Primary    │───▶│  Replica 1  │   │  Replica 2  │
│  (Write)    │    │  (Read)     │   │  (Read)     │
└─────────────┘    └─────────────┘   └─────────────┘
                         │                  │
                         └──────────┬───────┘
                                    │
                          Read Load Balancing
```

**Implementation:**
```python
# database/database.py - Read/Write splitting
from sqlalchemy import create_engine
from random import choice

PRIMARY_URL = os.getenv("DATABASE_URL")
REPLICA_URLS = os.getenv("DATABASE_REPLICAS", "").split(",")

primary_engine = create_engine(PRIMARY_URL)
replica_engines = [create_engine(url) for url in REPLICA_URLS if url]

def get_read_engine():
    """Load balance across read replicas."""
    if replica_engines:
        return choice(replica_engines)
    return primary_engine

def get_write_engine():
    """Always use primary for writes."""
    return primary_engine

# Usage
SessionLocalRead = sessionmaker(bind=get_read_engine())
SessionLocalWrite = sessionmaker(bind=get_write_engine())
```

### 1.3 Computational Bottlenecks

#### Issue: Synchronous Analysis Processing

**Current:** Analysis runs in request thread
```python
# api/main.py
@app.post("/analyze/nfw")
async def analyze_nfw(request: NFWRequest):
    # This blocks the worker for 5-30 seconds!
    model = load_pretrained_model()
    data = prepare_model_input(request.parameters)
    result = model(data)  # CPU-intensive
    
    return result
```

**Problem:**
- Request thread blocked during computation
- Other requests queued
- Poor user experience (long wait)
- Worker saturation

**Solution 1: Background Tasks with Celery**
```python
# tasks/analysis_tasks.py
from celery import Celery

celery_app = Celery(
    'lensing',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/1'
)

@celery_app.task
def run_nfw_analysis(analysis_id: int, parameters: dict):
    """Run NFW analysis asynchronously."""
    model = load_pretrained_model()
    data = prepare_model_input(parameters)
    result = model(data)
    
    # Update database with results
    db = SessionLocal()
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    analysis.status = "completed"
    analysis.result = result
    db.commit()
    
    return result

# api/main.py
@app.post("/analyze/nfw")
async def analyze_nfw(
    request: NFWRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    # Create analysis record
    analysis = Analysis(
        user_id=current_user.id,
        analysis_type="nfw",
        status="pending",
        parameters=request.dict()
    )
    db.add(analysis)
    db.commit()
    
    # Queue background task
    task = run_nfw_analysis.delay(analysis.id, request.dict())
    
    # Return immediately with job ID
    return {
        "analysis_id": analysis.id,
        "status": "pending",
        "task_id": task.id,
        "estimated_time": "5-10 minutes"
    }

@app.get("/analyze/{analysis_id}/status")
async def get_analysis_status(analysis_id: int):
    """Check analysis status."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    return {
        "status": analysis.status,
        "progress": analysis.progress,
        "result": analysis.result if analysis.status == "completed" else None
    }
```

**Solution 2: WebSocket for Real-time Updates**
```python
# api/websocket.py
from fastapi import WebSocket

@app.websocket("/ws/analysis/{analysis_id}")
async def analysis_updates(websocket: WebSocket, analysis_id: int):
    await websocket.accept()
    
    while True:
        # Check analysis status
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        
        # Send update to client
        await websocket.send_json({
            "status": analysis.status,
            "progress": analysis.progress,
            "message": f"Processing: {analysis.progress}%"
        })
        
        if analysis.status in ["completed", "failed"]:
            await websocket.close()
            break
        
        await asyncio.sleep(2)  # Update every 2 seconds
```

#### Issue: No Result Caching

**Problem:** Identical analyses recomputed every time.

**Example:**
```
User A: Analyze NFW profile with mass=1e14, concentration=5
  → Computation time: 10 seconds
  
User B: Analyze SAME NFW profile
  → Computation time: 10 seconds (redundant!)
```

**Solution: Redis Caching**
```python
import hashlib
import json
import redis

redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

def get_cache_key(analysis_type: str, parameters: dict) -> str:
    """Generate deterministic cache key."""
    param_str = json.dumps(parameters, sort_keys=True)
    return f"analysis:{analysis_type}:{hashlib.md5(param_str.encode()).hexdigest()}"

@app.post("/analyze/nfw")
async def analyze_nfw(request: NFWRequest):
    # Check cache
    cache_key = get_cache_key("nfw", request.dict())
    cached_result = redis_client.get(cache_key)
    
    if cached_result:
        logger.info(f"Cache hit: {cache_key}")
        return json.loads(cached_result)
    
    # Compute
    result = run_analysis(request)
    
    # Cache for 24 hours
    redis_client.setex(cache_key, 86400, json.dumps(result))
    
    return result
```

**Cache Performance:**
```
Without Caching:
├─ Request 1: 10 seconds
├─ Request 2 (same params): 10 seconds
├─ Request 3 (same params): 10 seconds
└─ Total: 30 seconds

With Caching:
├─ Request 1: 10 seconds (cache miss)
├─ Request 2 (same params): 5ms (cache hit, 2000x faster)
├─ Request 3 (same params): 5ms (cache hit)
└─ Total: 10.01 seconds (3x improvement)
```

### 1.4 File Storage Bottlenecks

#### Issue: Local Filesystem Storage

**Current:**
```yaml
# docker-compose.yml
volumes:
  - ./results:/app/results
  - ./data:/app/data
```

**Problems:**
1. No horizontal scaling (files on single host)
2. No redundancy (disk failure = data loss)
3. No CDN acceleration
4. No access control at storage layer

**Solution: Cloud Object Storage (S3-compatible)**

```python
# src/utils/storage.py
import boto3
from botocore.exceptions import ClientError

class CloudStorage:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('S3_ENDPOINT'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.bucket = os.getenv('S3_BUCKET', 'lensing-results')
    
    def upload_result(self, analysis_id: int, data: bytes, filename: str):
        """Upload analysis result to S3."""
        key = f"analyses/{analysis_id}/{filename}"
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType='application/octet-stream',
            ServerSideEncryption='AES256'
        )
        
        return key
    
    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str:
        """Generate temporary download URL."""
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expiration
        )
    
    def download_result(self, key: str) -> bytes:
        """Download result from S3."""
        response = self.s3.get_object(Bucket=self.bucket, Key=key)
        return response['Body'].read()

# Usage in API
storage = CloudStorage()

@app.post("/analyze/nfw")
async def analyze_nfw(request: NFWRequest):
    # Run analysis
    result = run_analysis(request)
    
    # Save to S3 instead of local disk
    result_bytes = json.dumps(result).encode()
    key = storage.upload_result(analysis_id, result_bytes, "result.json")
    
    # Generate temporary download URL
    download_url = storage.generate_presigned_url(key, expiration=7200)
    
    return {
        "analysis_id": analysis_id,
        "download_url": download_url,
        "expires_in": 7200
    }
```

**Benefits:**
- ✅ Horizontal scaling (distributed storage)
- ✅ High availability (99.999999999% durability)
- ✅ CDN integration (CloudFront, CloudFlare)
- ✅ Access control (presigned URLs)
- ✅ Encryption at rest
- ✅ Versioning support
- ✅ Lifecycle policies (auto-delete old results)

---

## 2. Load Testing & Benchmarking

### 2.1 Load Testing Strategy

**Tool: Locust**
```python
# load_tests/locustfile.py
from locust import HttpUser, task, between

class LensingUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login before tests."""
        response = self.client.post("/auth/login", json={
            "username": "test@example.com",
            "password": "testpass"
        })
        self.token = response.json()["access_token"]
        self.client.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def list_analyses(self):
        """Most common operation - view analyses."""
        self.client.get("/api/analyses")
    
    @task(1)
    def create_analysis(self):
        """Create new NFW analysis."""
        self.client.post("/api/analyze/nfw", json={
            "mass": 1e14,
            "concentration": 5,
            "grid_size": 64
        })
    
    @task(2)
    def get_analysis_result(self):
        """Check analysis status."""
        self.client.get("/api/analyses/123/status")
```

**Run Tests:**
```bash
# Install locust
pip install locust

# Run load test
locust -f load_tests/locustfile.py --host=http://localhost:8000

# Command-line test
locust -f load_tests/locustfile.py \
    --host=http://localhost:8000 \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --headless
```

### 2.2 Expected Performance Bottlenecks

**Scenario 1: 100 Concurrent Users**
```
Current Architecture (No optimizations):
├─ API Throughput: ~10 req/s
├─ Database Connections: Exhausted (default 20 pool)
├─ CPU Usage: 100% (saturated)
├─ Memory Usage: 60% (2.4 GB / 4 GB)
├─ Response Time P95: 15-20 seconds
└─ Error Rate: 5-10% (timeouts, connection errors)

Bottlenecks:
1. Single Uvicorn worker (CPU bound)
2. Database connection pool exhausted
3. Synchronous analysis processing
4. No caching (redundant computations)
```

**Scenario 2: 1,000 Concurrent Users**
```
Current Architecture:
├─ API: ❌ FAILURE - Service unavailable
├─ Database: ❌ FAILURE - Max connections exceeded
├─ Memory: ❌ OOM errors
└─ Impact: Complete system outage

Required Changes:
1. Horizontal scaling (multiple API instances)
2. Load balancer
3. Database connection pooling + read replicas
4. Async processing with Celery
5. Redis caching layer
6. CDN for static assets
```

### 2.3 Performance Benchmarks (Target vs Current)

| Metric | Current (Estimated) | Target | Optimization Required |
|--------|---------------------|--------|----------------------|
| **API Response Time (p50)** | 2-5s | <100ms | Multi-worker, caching |
| **API Response Time (p95)** | 10-20s | <500ms | Async processing |
| **Throughput** | ~10 req/s | 1,000+ req/s | Horizontal scaling |
| **Concurrent Users** | ~10 | 10,000+ | Load balancing, HA |
| **Database Query Time** | 100-500ms | <10ms | Indexing, query optimization |
| **Analysis Processing** | 10-30s (blocking) | <1 min (async) | Celery, GPU |
| **Cache Hit Rate** | 0% (no cache) | >80% | Redis implementation |

---

## 3. Horizontal Scaling Architecture

### 3.1 Target Architecture (Cloud-Native)

```
                     ┌──────────────────┐
                     │   Load Balancer  │
                     │  (AWS ALB/ELB)   │
                     └────────┬─────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
        ┌─────────────┐┌─────────────┐┌─────────────┐
        │  API Pod 1  ││  API Pod 2  ││  API Pod N  │
        │  (FastAPI)  ││  (FastAPI)  ││  (FastAPI)  │
        └──────┬──────┘└──────┬──────┘└──────┬──────┘
               │               │               │
               └───────────────┼───────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
       ┌──────────────┐┌─────────────┐┌────────────────┐
       │  PostgreSQL  ││    Redis    ││  Celery Workers│
       │  (Primary +  ││  (Cluster)  ││  (Auto-scale)  │
       │  Replicas)   ││             ││                │
       └──────────────┘└─────────────┘└────────────────┘
                │              │              │
                └──────────────┼──────────────┘
                               │
                        ┌──────┴──────┐
                        │     S3      │
                        │  (Storage)  │
                        └─────────────┘
```

### 3.2 Kubernetes Deployment

**Deployment Manifest:**
```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lensing-api
  labels:
    app: lensing
    component: api
spec:
  replicas: 3  # Start with 3 pods
  selector:
    matchLabels:
      app: lensing
      component: api
  template:
    metadata:
      labels:
        app: lensing
        component: api
    spec:
      containers:
      - name: api
        image: lensing-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: lensing-secrets
              key: database-url
        - name: REDIS_URL
          value: redis://lensing-redis:6379
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: lensing-api
spec:
  selector:
    app: lensing
    component: api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Horizontal Pod Autoscaler:**
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lensing-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lensing-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max
```

### 3.3 Load Balancing Strategy

**AWS Application Load Balancer (ALB) Configuration:**
```yaml
# terraform/alb.tf
resource "aws_lb" "lensing" {
  name               = "lensing-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnets

  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true

  tags = {
    Environment = "production"
  }
}

resource "aws_lb_target_group" "api" {
  name     = "lensing-api-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = var.vpc_id

  health_check {
    enabled             = true
    path                = "/health"
    port                = "traffic-port"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }

  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }

  deregistration_delay = 30
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.lensing.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = var.ssl_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
```

---

## 4. Database Scaling Strategy

### 4.1 Read Replica Configuration

**PostgreSQL Replication:**
```yaml
# docker-compose-ha.yml
services:
  postgres-primary:
    image: postgres:15
    environment:
      POSTGRES_USER: lensing
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: lensing_db
    command: >
      postgres
      -c wal_level=replica
      -c max_wal_senders=10
      -c max_replication_slots=10
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
  
  postgres-replica-1:
    image: postgres:15
    environment:
      PGUSER: replicator
      PGPASSWORD: ${REPLICATION_PASSWORD}
    command: >
      bash -c "
      until pg_basebackup --pgdata=/var/lib/postgresql/data -R --slot=replication_slot --host=postgres-primary --port=5432
      do
        echo 'Waiting for primary to be ready...'
        sleep 1s
      done
      postgres
      "
    depends_on:
      - postgres-primary
    volumes:
      - postgres_replica_1_data:/var/lib/postgresql/data
  
  postgres-replica-2:
    image: postgres:15
    # Same as replica-1
```

### 4.2 Connection Pooling with PgBouncer

```yaml
# docker-compose-ha.yml
services:
  pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      DATABASES: lensing_db=host=postgres-primary port=5432 dbname=lensing_db
      PGBOUNCER_POOL_MODE: transaction
      PGBOUNCER_MAX_CLIENT_CONN: 1000
      PGBOUNCER_DEFAULT_POOL_SIZE: 25
      PGBOUNCER_MIN_POOL_SIZE: 5
      PGBOUNCER_RESERVE_POOL_SIZE: 5
    ports:
      - "6432:6432"
    depends_on:
      - postgres-primary
```

**Update Application:**
```python
# Use PgBouncer instead of direct connection
DATABASE_URL = "postgresql://lensing:pass@pgbouncer:6432/lensing_db"
```

### 4.3 Query Performance Monitoring

```python
# database/monitoring.py
from sqlalchemy import event
from sqlalchemy.engine import Engine
import logging
import time

logger = logging.getLogger("sql_performance")

@event.listens_for(Engine, "before_cursor_execute")
def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total_time = time.time() - conn.info['query_start_time'].pop(-1)
    
    # Log slow queries (>100ms)
    if total_time > 0.1:
        logger.warning(f"Slow query ({total_time:.3f}s): {statement[:100]}...")
    
    # Track metrics
    metrics.histogram('database.query.duration', total_time)
    metrics.increment('database.query.count')
```

---

## 5. Caching Strategy

### 5.1 Multi-Layer Caching

```
┌─────────────────────────────────────────┐
│         Application Layer                │
│  ┌────────────────────────────────────┐ │
│  │  L1: In-Memory Cache (LRU)         │ │
│  │  - Model weights                    │ │
│  │  - Frequently accessed configs      │ │
│  │  - TTL: 5 minutes                   │ │
│  └────────────────────────────────────┘ │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         Redis Layer (L2)                 │
│  ┌────────────────────────────────────┐ │
│  │  - Analysis results                 │ │
│  │  - User sessions                    │ │
│  │  - API responses                    │ │
│  │  - TTL: 24 hours                    │ │
│  └────────────────────────────────────┘ │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│         Database (L3)                    │
│  - Persistent storage                    │
│  - No TTL                                │
└─────────────────────────────────────────┘
```

**Implementation:**
```python
# src/utils/cache.py
from functools import wraps
import redis
import pickle
from typing import Any, Callable

redis_client = redis.Redis(host='redis', port=6379)

def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results in Redis."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            cache_key = f"{key_prefix}:{func.__name__}:{args}:{kwargs}"
            
            # Check cache
            cached_value = redis_client.get(cache_key)
            if cached_value:
                return pickle.loads(cached_value)
            
            # Compute value
            result = func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(cache_key, ttl, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cached(ttl=86400, key_prefix="analysis")
def run_nfw_analysis(mass: float, concentration: float):
    # Expensive computation
    return compute_convergence(mass, concentration)
```

---

## 6. Performance Monitoring

### 6.1 Application Performance Monitoring (APM)

**Add New Relic / DataDog:**
```python
# api/main.py
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

app = FastAPI()

@app.middleware("http")
async def add_apm_tracing(request: Request, call_next):
    with newrelic.agent.FunctionTrace(name=f"{request.method} {request.url.path}"):
        response = await call_next(request)
        return response
```

### 6.2 Custom Metrics

```python
# src/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
analysis_requests = Counter('analysis_requests_total', 'Total analysis requests')
analysis_duration = Histogram('analysis_duration_seconds', 'Analysis execution time')
active_analyses = Gauge('active_analyses', 'Number of active analyses')

# Use in code
@analysis_duration.time()
def run_analysis(params):
    analysis_requests.inc()
    active_analyses.inc()
    try:
        # Run analysis
        result = compute(params)
        return result
    finally:
        active_analyses.dec()
```

---

## Summary: Critical Optimizations

### Phase 1: Immediate (1 week)
1. ✅ Add Gunicorn with 4 workers
2. ✅ Implement database indexes
3. ✅ Add Redis result caching
4. ✅ Configure connection pooling
5. ✅ Add request timeouts

**Expected Impact:** 3-5x throughput improvement

### Phase 2: Short-term (1 month)
1. ✅ Implement Celery for async processing
2. ✅ Add database read replicas
3. ✅ Implement S3 storage
4. ✅ Add load testing suite
5. ✅ Configure monitoring dashboards

**Expected Impact:** 10-20x throughput improvement

### Phase 3: Long-term (3 months)
1. ✅ Kubernetes deployment
2. ✅ Horizontal autoscaling
3. ✅ CDN integration
4. ✅ Multi-region deployment
5. ✅ GPU acceleration for PINN

**Expected Impact:** 100x+ throughput improvement

---

**Next Document:** See `CLOUD_NATIVE_ROADMAP.md` for cloud deployment strategy.
