# Cloud-Native Deployment Roadmap

**Document Version:** 1.0  
**Date:** October 7, 2025  
**Focus:** Kubernetes, Cloud Services, Infrastructure as Code  
**Target:** Production-grade, auto-scaling, multi-region deployment

---

## Executive Summary

### Current State
- **Deployment:** Docker Compose (single host)
- **Orchestration:** None
- **Scaling:** Manual, vertical only
- **High Availability:** None (single point of failure)
- **Disaster Recovery:** Manual backups only

### Target State
- **Deployment:** Kubernetes on AWS/GCP/Azure
- **Orchestration:** K8s + Helm charts
- **Scaling:** Auto-scaling (horizontal + vertical)
- **High Availability:** Multi-AZ with 99.9% uptime SLA
- **Disaster Recovery:** Automated backups, <15 min RTO

### Migration Timeline
- **Phase 1:** Kubernetes setup (2 weeks)
- **Phase 2:** Cloud services integration (3 weeks)
- **Phase 3:** Production hardening (2 weeks)
- **Phase 4:** Multi-region deployment (4 weeks)

**Total Timeline:** 11 weeks (3 months)

---

## 1. Kubernetes Architecture

### 1.1 Cluster Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Kubernetes Cluster                            │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    Ingress Nginx                            │    │
│  │  (Load Balancer + SSL Termination + Rate Limiting)         │    │
│  └──────────────────────┬──────────────────────────────────────┘    │
│                         │                                             │
│  ┌──────────────────────┼──────────────────────────────────────┐    │
│  │          Namespaces  │                                       │    │
│  │  ┌───────────────────▼────────────┐  ┌───────────────────┐ │    │
│  │  │  Namespace: production         │  │  Namespace: stage │ │    │
│  │  │                                │  │                   │ │    │
│  │  │  ┌──────────────────────────┐ │  │  ┌─────────────┐ │ │    │
│  │  │  │   API Deployment (HPA)   │ │  │  │ API (dev)   │ │ │    │
│  │  │  │   Replicas: 3-20         │ │  │  │ Replica: 1  │ │ │    │
│  │  │  │   - lensing-api:latest   │ │  │  └─────────────┘ │ │    │
│  │  │  └──────────────────────────┘ │  │                   │ │    │
│  │  │                                │  └───────────────────┘ │    │
│  │  │  ┌──────────────────────────┐ │                         │    │
│  │  │  │  Celery Workers (HPA)    │ │                         │    │
│  │  │  │  Replicas: 2-10          │ │                         │    │
│  │  │  │  - GPU-enabled pods      │ │                         │    │
│  │  │  └──────────────────────────┘ │                         │    │
│  │  │                                │                         │    │
│  │  │  ┌──────────────────────────┐ │                         │    │
│  │  │  │  Streamlit Web App       │ │                         │    │
│  │  │  │  Replicas: 2             │ │                         │    │
│  │  │  └──────────────────────────┘ │                         │    │
│  │  └────────────────────────────────┘                         │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   StatefulSets                               │    │
│  │  ┌─────────────────────┐  ┌────────────────────────────┐   │    │
│  │  │  Redis Cluster      │  │  PostgreSQL (via Operator) │   │    │
│  │  │  - 3 masters        │  │  - Primary + 2 Replicas    │   │    │
│  │  │  - 3 replicas       │  │  - Auto-failover           │   │    │
│  │  └─────────────────────┘  └────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   Persistent Storage                         │    │
│  │  ┌─────────────────────┐  ┌────────────────────────────┐   │    │
│  │  │  AWS EBS (Database) │  │  AWS EFS (Shared Files)    │   │    │
│  │  │  - gp3 SSD          │  │  - Multi-AZ                │   │    │
│  │  │  - Snapshots enabled│  │  - Encryption enabled      │   │    │
│  │  └─────────────────────┘  └────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                   Observability                              │    │
│  │  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │    │
│  │  │  Prometheus  │  │  Grafana    │  │  ELK Stack      │   │    │
│  │  │  (Metrics)   │  │  (Dashboar  │  │  (Logs)         │   │    │
│  │  └──────────────┘  └─────────────┘  └─────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

### 1.2 Infrastructure as Code (Terraform)

**Project Structure:**
```
terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── providers.tf            # Provider configuration
├── versions.tf             # Version constraints
├── modules/
│   ├── eks-cluster/        # EKS cluster module
│   ├── rds-postgres/       # RDS database module
│   ├── elasticache-redis/  # Redis cluster module
│   ├── s3-storage/         # S3 bucket module
│   ├── vpc-networking/     # VPC and networking
│   └── monitoring/         # CloudWatch, Prometheus
└── environments/
    ├── dev/
    │   └── terraform.tfvars
    ├── staging/
    │   └── terraform.tfvars
    └── production/
        └── terraform.tfvars
```

**Main Configuration:**
```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  
  backend "s3" {
    bucket         = "lensing-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "gravitational-lensing"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# VPC and Networking
module "vpc" {
  source = "./modules/vpc-networking"
  
  vpc_cidr           = var.vpc_cidr
  availability_zones = var.availability_zones
  environment        = var.environment
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
}

# EKS Cluster
module "eks" {
  source = "./modules/eks-cluster"
  
  cluster_name    = "lensing-${var.environment}"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnet_ids
  
  node_groups = {
    general = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 2
      instance_types   = ["t3.xlarge"]
      disk_size        = 100
    }
    
    gpu_compute = {
      desired_capacity = 1
      max_capacity     = 5
      min_capacity     = 0
      instance_types   = ["g4dn.xlarge"]  # NVIDIA T4 GPU
      disk_size        = 200
      labels = {
        workload = "gpu"
      }
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NoSchedule"
      }]
    }
  }
}

# RDS PostgreSQL
module "rds" {
  source = "./modules/rds-postgres"
  
  identifier     = "lensing-${var.environment}"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  
  multi_az               = var.environment == "production"
  backup_retention_period = var.environment == "production" ? 30 : 7
  
  vpc_security_group_ids = [module.vpc.database_security_group_id]
  db_subnet_group_name   = module.vpc.database_subnet_group_name
  
  database_name = "lensing_db"
  username      = var.db_username
  password      = var.db_password  # Use AWS Secrets Manager in production
  
  performance_insights_enabled = true
  monitoring_interval          = 60
  
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
}

# ElastiCache Redis
module "redis" {
  source = "./modules/elasticache-redis"
  
  cluster_id           = "lensing-${var.environment}"
  engine_version       = "7.0"
  node_type            = var.redis_node_type
  num_cache_nodes      = var.environment == "production" ? 3 : 1
  
  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled          = var.environment == "production"
  
  subnet_group_name    = module.vpc.elasticache_subnet_group_name
  security_group_ids   = [module.vpc.redis_security_group_id]
  
  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"
  maintenance_window      = "sun:05:00-sun:07:00"
}

# S3 Storage
module "s3" {
  source = "./modules/s3-storage"
  
  bucket_name = "lensing-${var.environment}-results"
  
  versioning_enabled = var.environment == "production"
  lifecycle_rules = [
    {
      id      = "archive-old-results"
      enabled = true
      transitions = [
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]
      expiration = {
        days = 365
      }
    }
  ]
  
  cors_rules = [
    {
      allowed_origins = ["https://*.yourdomain.com"]
      allowed_methods = ["GET", "PUT", "POST"]
      allowed_headers = ["*"]
      max_age_seconds = 3000
    }
  ]
  
  server_side_encryption = {
    sse_algorithm = "aws:kms"
  }
}

# CloudFront CDN
resource "aws_cloudfront_distribution" "cdn" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Lensing Results CDN"
  default_root_object = "index.html"
  
  origin {
    domain_name = module.s3.bucket_regional_domain_name
    origin_id   = "S3-lensing-results"
    
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.oai.cloudfront_access_identity_path
    }
  }
  
  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = "S3-lensing-results"
    
    forwarded_values {
      query_string = false
      cookies {
        forward = "none"
      }
    }
    
    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
  }
  
  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }
  
  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
```

**Variables:**
```hcl
# terraform/variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.large"
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.medium"
}
```

**Production Values:**
```hcl
# terraform/environments/production/terraform.tfvars
aws_region         = "us-east-1"
environment        = "production"
vpc_cidr           = "10.0.0.0/16"
availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]

# Database
db_instance_class = "db.r6g.xlarge"  # 4 vCPU, 32 GB RAM
db_username       = "lensing_admin"

# Cache
redis_node_type = "cache.r6g.large"  # 2 vCPU, 13 GB RAM

# EKS
eks_node_instance_types = ["t3.2xlarge"]  # 8 vCPU, 32 GB RAM
eks_desired_capacity    = 5
eks_max_capacity        = 20
```

### 1.3 Kubernetes Manifests

**Namespace:**
```yaml
# k8s/namespaces/production.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lensing-production
  labels:
    name: lensing-production
    environment: production
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: compute-quota
  namespace: lensing-production
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    limits.cpu: "200"
    limits.memory: 400Gi
    persistentvolumeclaims: "10"
```

**API Deployment:**
```yaml
# k8s/deployments/api.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lensing-api
  namespace: lensing-production
  labels:
    app: lensing
    component: api
    version: v2.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: lensing
      component: api
  template:
    metadata:
      labels:
        app: lensing
        component: api
        version: v2.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: lensing-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      
      initContainers:
      - name: wait-for-db
        image: busybox:latest
        command: ['sh', '-c', 'until nc -z postgres 5432; do echo waiting for postgres; sleep 2; done;']
      
      - name: run-migrations
        image: lensing-api:latest
        command: ['alembic', 'upgrade', 'head']
        envFrom:
        - secretRef:
            name: lensing-secrets
      
      containers:
      - name: api
        image: lensing-api:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "info"
        - name: WORKERS
          value: "4"
        
        envFrom:
        - secretRef:
            name: lensing-secrets
        - configMapRef:
            name: lensing-config
        
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
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]
        
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/.cache
      
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: component
                  operator: In
                  values:
                  - api
              topologyKey: kubernetes.io/hostname
      
      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: DoNotSchedule
        labelSelector:
          matchLabels:
            component: api
---
apiVersion: v1
kind: Service
metadata:
  name: lensing-api
  namespace: lensing-production
  labels:
    app: lensing
    component: api
spec:
  type: ClusterIP
  selector:
    app: lensing
    component: api
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lensing-api-hpa
  namespace: lensing-production
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
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
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
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: lensing-api-pdb
  namespace: lensing-production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: lensing
      component: api
```

**Celery Worker Deployment:**
```yaml
# k8s/deployments/celery-worker.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lensing-celery-worker
  namespace: lensing-production
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lensing
      component: celery-worker
  template:
    metadata:
      labels:
        app: lensing
        component: celery-worker
    spec:
      containers:
      - name: worker
        image: lensing-api:latest
        command: ['celery', '-A', 'tasks.celery_app', 'worker', '--loglevel=info', '--concurrency=4']
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1  # Request GPU
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        
        envFrom:
        - secretRef:
            name: lensing-secrets
        - configMapRef:
            name: lensing-config
      
      nodeSelector:
        workload: gpu
      
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: lensing-celery-worker-hpa
  namespace: lensing-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: lensing-celery-worker
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_length
      target:
        type: AverageValue
        averageValue: "10"
```

**Ingress:**
```yaml
# k8s/ingress/api-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: lensing-ingress
  namespace: lensing-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "10"
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://yourdomain.com"
spec:
  tls:
  - hosts:
    - api.lensing.yourdomain.com
    secretName: lensing-api-tls
  rules:
  - host: api.lensing.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: lensing-api
            port:
              number: 80
```

---

## 2. Helm Charts

### 2.1 Chart Structure

```
helm/lensing/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-staging.yaml
├── values-production.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment-api.yaml
│   ├── deployment-celery.yaml
│   ├── deployment-webapp.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── serviceaccount.yaml
│   └── tests/
│       └── test-connection.yaml
└── charts/
    ├── postgresql/  # Dependency
    └── redis/       # Dependency
```

**Chart.yaml:**
```yaml
apiVersion: v2
name: lensing
description: Gravitational Lensing Analysis Platform
type: application
version: 2.0.0
appVersion: "2.0.0"

keywords:
  - gravitational-lensing
  - astronomy
  - machine-learning
  - pinn

home: https://github.com/yourorg/lensing
sources:
  - https://github.com/yourorg/lensing

maintainers:
  - name: Your Team
    email: team@yourdomain.com

dependencies:
  - name: postgresql
    version: "12.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: postgresql.enabled
  
  - name: redis
    version: "17.x.x"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
```

**values.yaml:**
```yaml
# Default values for lensing
replicaCount: 3

image:
  repository: yourregistry/lensing-api
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8000"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: api.lensing.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: lensing-api-tls
      hosts:
        - api.lensing.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app
                operator: In
                values:
                  - lensing
          topologyKey: kubernetes.io/hostname

# Celery workers
celeryWorker:
  enabled: true
  replicaCount: 2
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi
      nvidia.com/gpu: 1
    requests:
      cpu: 2000m
      memory: 4Gi
      nvidia.com/gpu: 1
  nodeSelector:
    workload: gpu
  tolerations:
    - key: nvidia.com/gpu
      operator: Equal
      value: "true"
      effect: NoSchedule

# PostgreSQL (managed externally in production)
postgresql:
  enabled: false
  external:
    host: lensing-prod.xxx.us-east-1.rds.amazonaws.com
    port: 5432
    database: lensing_db
    existingSecret: lensing-db-credentials

# Redis (managed externally in production)
redis:
  enabled: false
  external:
    host: lensing-prod.xxx.cache.amazonaws.com
    port: 6379

# Application configuration
config:
  logLevel: info
  environment: production
  workers: 4
  
  # S3 Storage
  s3:
    bucket: lensing-production-results
    region: us-east-1
  
  # Feature flags
  features:
    gpuAcceleration: true
    caching: true
    monitoring: true

# Secrets (override in each environment)
secrets:
  databaseUrl: ""
  redisUrl: ""
  secretKey: ""
  awsAccessKeyId: ""
  awsSecretAccessKey: ""
```

**Deploy with Helm:**
```bash
# Add dependencies
helm dependency update helm/lensing

# Install
helm install lensing ./helm/lensing \
  --namespace lensing-production \
  --create-namespace \
  --values helm/lensing/values-production.yaml \
  --wait --timeout 10m

# Upgrade
helm upgrade lensing ./helm/lensing \
  --namespace lensing-production \
  --values helm/lensing/values-production.yaml \
  --wait --timeout 10m

# Rollback
helm rollback lensing -n lensing-production
```

---

## 3. CI/CD Pipeline for Kubernetes

### 3.1 GitHub Actions Workflow

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags:
      - 'v*'
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  EKS_CLUSTER: lensing-production
  ECR_REPOSITORY: lensing-api

jobs:
  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v2
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ steps.login-ecr.outputs.registry }}/${{ env.ECR_REPOSITORY }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
  
  deploy-to-k8s:
    name: Deploy to Kubernetes
    needs: build-and-push
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://api.lensing.yourdomain.com
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    
    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig \
          --region ${{ env.AWS_REGION }} \
          --name ${{ env.EKS_CLUSTER }}
    
    - name: Install Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
    
    - name: Deploy with Helm
      run: |
        helm upgrade --install lensing ./helm/lensing \
          --namespace lensing-production \
          --create-namespace \
          --values helm/lensing/values-production.yaml \
          --set image.tag=${{ needs.build-and-push.outputs.image-tag }} \
          --wait --timeout 10m
    
    - name: Verify deployment
      run: |
        kubectl rollout status deployment/lensing-api \
          -n lensing-production --timeout=5m
    
    - name: Run smoke tests
      run: |
        kubectl run smoke-test --rm -i --restart=Never \
          --namespace=lensing-production \
          --image=curlimages/curl:latest \
          -- curl -f http://lensing-api/health || exit 1
    
    - name: Notify Slack
      if: always()
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'Production deployment ${{ job.status }}'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 4. Monitoring and Observability

### 4.1 Prometheus Setup

```yaml
# k8s/monitoring/prometheus-values.yaml
prometheus:
  prometheusSpec:
    retention: 30d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 100Gi
    
    resources:
      requests:
        cpu: 2000m
        memory: 4Gi
      limits:
        cpu: 4000m
        memory: 8Gi
    
    serviceMonitorSelector:
      matchLabels:
        prometheus: kube-prometheus
    
    additionalScrapeConfigs:
      - job_name: 'lensing-api'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - lensing-production
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__

grafana:
  adminPassword: changeme
  
  dashboards:
    default:
      lensing-api:
        gnetId: 1860  # Node Exporter Full
        revision: 27
        datasource: Prometheus
      
      lensing-custom:
        url: https://grafana.com/api/dashboards/12345/revisions/1/download
  
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: Prometheus
          type: prometheus
          url: http://prometheus-server
          isDefault: true
```

**Install Prometheus:**
```bash
# Add Prometheus community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values k8s/monitoring/prometheus-values.yaml
```

### 4.2 Custom Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Lensing API Performance",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{job=\"lensing-api\"}[5m])"
          }
        ]
      },
      {
        "title": "Request Latency (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "Active Analyses",
        "targets": [
          {
            "expr": "active_analyses"
          }
        ]
      },
      {
        "title": "Celery Queue Length",
        "targets": [
          {
            "expr": "celery_queue_length"
          }
        ]
      }
    ]
  }
}
```

---

## 5. Cost Optimization

### 5.1 Resource Right-Sizing

**Cost Analysis:**
```
Production Environment (Monthly Costs):

EKS Cluster Control Plane:
  └─ $73/month (flat fee)

EC2 Instances (General Node Group):
  ├─ 3x t3.xlarge (4 vCPU, 16 GB): $0.1664/hour
  └─ 3 * $0.1664 * 730 hours = $365/month

EC2 Instances (GPU Node Group - spot):
  ├─ 1x g4dn.xlarge (4 vCPU, 16 GB, T4 GPU): $0.526/hour
  ├─ Spot discount: 70% ($0.158/hour)
  └─ 1 * $0.158 * 730 hours = $115/month

RDS PostgreSQL (db.r6g.xlarge):
  ├─ On-Demand: $0.336/hour
  ├─ Reserved 1-year: $0.212/hour (37% savings)
  └─ $0.212 * 730 hours = $155/month

ElastiCache Redis (cache.r6g.large):
  ├─ On-Demand: $0.201/hour
  └─ $0.201 * 730 hours = $147/month

S3 Storage (10 TB):
  ├─ Standard: $0.023/GB = $230/month
  ├─ Glacier (after 90 days): $0.004/GB = $40/month
  └─ Average: $135/month

Data Transfer:
  ├─ Outbound: 5 TB * $0.09/GB = $450/month
  └─ CloudFront: 5 TB * $0.085/GB = $425/month (save $25)

Total Monthly Cost: ~$1,415/month
Annual Cost: ~$17,000/year

With Optimizations (Reserved, Spot, Glacier):
  └─ ~$1,100/month (~$13,200/year) - 22% savings
```

### 5.2 Cost Optimization Strategies

1. **Use Spot Instances for GPU Workers**
```yaml
# terraform/modules/eks-cluster/main.tf
resource "aws_eks_node_group" "gpu_spot" {
  capacity_type = "SPOT"
  instance_types = ["g4dn.xlarge", "g4dn.2xlarge"]
  # 70% cost savings vs on-demand
}
```

2. **Enable Cluster Autoscaler**
```yaml
# k8s/monitoring/cluster-autoscaler.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-priority-expander
  namespace: kube-system
data:
  priorities: |-
    10:
      - .*-spot-.*
    50:
      - .*-on-demand-.*
```

3. **S3 Lifecycle Policies**
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "results" {
  rule {
    id = "archive-old-results"
    transitions {
      days          = 90
      storage_class = "GLACIER"  # 82% cost reduction
    }
    expiration {
      days = 365  # Delete after 1 year
    }
  }
}
```

4. **RDS Reserved Instances**
```bash
# Purchase 1-year reserved instance (37% savings)
aws rds purchase-reserved-db-instances-offering \
  --reserved-db-instances-offering-id xxx \
  --reserved-db-instance-id lensing-prod-ri
```

---

## 6. Disaster Recovery

### 6.1 Backup Strategy

**RDS Automated Backups:**
```hcl
# terraform/modules/rds-postgres/main.tf
resource "aws_db_instance" "main" {
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  copy_tags_to_snapshot  = true
  
  # Point-in-time recovery
  enabled_cloudwatch_logs_exports = ["postgresql"]
}
```

**Velero for Kubernetes Backup:**
```bash
# Install Velero
helm install velero vmware-tanzu/velero \
  --namespace velero \
  --set-file credentials.secretContents.cloud=./credentials-velero \
  --set configuration.provider=aws \
  --set configuration.backupStorageLocation.bucket=lensing-k8s-backups \
  --set configuration.backupStorageLocation.config.region=us-east-1 \
  --set snapshotsEnabled=true

# Create daily backup schedule
velero schedule create daily-backup \
  --schedule="@daily" \
  --include-namespaces lensing-production \
  --ttl 720h  # 30 days retention
```

### 6.2 Recovery Procedures

**Database Recovery:**
```bash
# Restore from automated backup
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier lensing-prod \
  --target-db-instance-identifier lensing-prod-restored \
  --restore-time 2025-10-07T12:00:00Z
```

**Kubernetes Recovery:**
```bash
# List backups
velero backup get

# Restore from backup
velero restore create --from-backup daily-backup-20251007
```

---

## 7. Multi-Region Deployment

### 7.1 Global Architecture

```
                    ┌────────────────────┐
                    │  Route 53 (DNS)    │
                    │  Latency Routing   │
                    └──────────┬─────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
        ┌──────────────┐┌──────────────┐┌──────────────┐
        │  us-east-1   ││  eu-west-1   ││  ap-south-1  │
        │  (Primary)   ││  (Secondary) ││  (Secondary) │
        └──────┬───────┘└──────┬───────┘└──────┬───────┘
               │                │                │
               └────────────────┼────────────────┘
                                │
                         ┌──────┴──────┐
                         │  RDS Global │
                         │  Database   │
                         │  (Primary + │
                         │  Replicas)  │
                         └─────────────┘
```

**Total Timeline:** 11 weeks (3 months)  
**Estimated Cost:** $1,100-1,500/month (optimized)  
**Target SLA:** 99.9% uptime

---

**Next Document:** See `REALTIME_COLLABORATION.md` for WebSocket and collaboration features.
