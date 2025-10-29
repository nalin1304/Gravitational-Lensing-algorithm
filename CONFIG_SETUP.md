# 🔧 Configuration Setup Guide

## ⚠️ ACTION REQUIRED: Items That Need Your Configuration

This document lists **ALL** items that require your manual configuration before deployment.

---

## 📋 Quick Checklist

- [ ] **Environment Variables** (.env file)
- [ ] **Database Credentials** (PostgreSQL)
- [ ] **Docker Hub Account** (for CI/CD) - [See DOCKER_SETUP.md](DOCKER_SETUP.md)
- [ ] **AWS Credentials** (for deployment)
- [ ] **API Keys** (external services)
- [ ] **Slack Webhook** (notifications)
- [ ] **Domain Configuration** (production URLs)

**🐳 NEW:** Detailed Docker setup guide available: [`DOCKER_SETUP.md`](DOCKER_SETUP.md)

---

## 1️⃣ Environment Variables (.env)

### 📍 Location: `d:\Coding projects\Collab\financial-advisor-tool\.env`

**STATUS:** 🔴 **NOT CONFIGURED** - File needs to be created

### Action Required:
Create a `.env` file in the project root with the following:

```bash
# ============================================
# DATABASE CONFIGURATION
# ============================================
# 🔴 TODO: Set your PostgreSQL credentials
DATABASE_URL=postgresql://USERNAME:PASSWORD@localhost:5432/lensing_db
POSTGRES_USER=lensing_user          # ← Change this
POSTGRES_PASSWORD=your_password     # ← Change this
POSTGRES_DB=lensing_db

# ============================================
# REDIS CONFIGURATION (Optional)
# ============================================
# 🟡 OPTIONAL: For caching and job queues
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=                     # ← Set if using password

# ============================================
# API CONFIGURATION
# ============================================
# 🟢 OK: Default values (change if needed)
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ============================================
# STREAMLIT CONFIGURATION
# ============================================
# 🟢 OK: Default values
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_THEME_BASE=dark
STREAMLIT_THEME_PRIMARY_COLOR=#FF4B4B

# ============================================
# SECURITY
# ============================================
# 🔴 TODO: Generate secure secret keys
SECRET_KEY=09234567890FGHEJKOORFJNFEKLFKN FDS@#$%^#&*()_+<>?7899JEFNDJNDNMKDNDMDCDMK DDVNNDVahfhoifuvhfjifvjfjekofd
JWT_SECRET=ghfiejfjkdojnfdjkefkjnfkeofjnfjkeofjnVNDKJVNFEKKFNEKOFNEKEKNF234567890-098765567890!@#$%^&*
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ============================================
# EXTERNAL API KEYS (if using)
# ============================================
# 🟡 OPTIONAL: Only if you're using external services
OPENAI_API_KEY=                     # For AI features (optional)
HUGGINGFACE_TOKEN=                  # For model downloads (optional)

# ============================================
# AWS CONFIGURATION (for deployment)
# ============================================
# 🟡 OPTIONAL: Only needed for cloud deployment
AWS_ACCESS_KEY_ID=                  # ← Set for AWS deployment
AWS_SECRET_ACCESS_KEY=              # ← Set for AWS deployment
AWS_REGION=us-east-1
S3_BUCKET_NAME=lensing-data

# ============================================
# MONITORING & LOGGING
# ============================================
# 🟢 OK: Default values
LOG_LEVEL=INFO
SENTRY_DSN=                         # ← Optional: For error tracking

# ============================================
# DEVELOPMENT vs PRODUCTION
# ============================================
# 🔴 TODO: Set to 'production' when deploying
ENVIRONMENT=development ⚠️ Error importing modules: No module named 'astropy'

Some features may be unavailable. Check that all dependencies are installed.

🔭 Gravitational Lensing Analysis Platform
Advanced Physics-Informed Neural Networks with Bayesian Uncertainty Quantification

Phase 15 Complete
⚠️ Required modules not available. Please check install            # ← Change to 'production' for deployment
DEBUG=True                          # ← Set to False in production
```

### 🔐 Generate Secure Secret Keys:

Run these commands to generate secure random keys:

```powershell
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 2️⃣ Database Setup

### 📍 PostgreSQL Installation

**STATUS:** 🔴 **REQUIRES INSTALLATION**

### Option A: Local PostgreSQL

1. **Install PostgreSQL:**
   ```powershell
   # Download from: https://www.postgresql.org/download/windows/
   # Or use Chocolatey:
   choco install postgresql
   ```

2. **Create Database:**
   ```sql
   -- Connect to PostgreSQL
   psql -U postgres
   
   -- Create user
   CREATE USER lensing_user WITH PASSWORD 'your_password_here';
   
   -- Create database
   CREATE DATABASE lensing_db OWNER lensing_user;
   
   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE lensing_db TO lensing_user;
   ```

3. **Update `.env` file** with your credentials

### Option B: Docker PostgreSQL (Recommended)

```powershell
# Run PostgreSQL in Docker
docker run -d \
  --name lensing-postgres \
  -e POSTGRES_USER=lensing_user \
  -e POSTGRES_PASSWORD=your_password_here \
  -e POSTGRES_DB=lensing_db \
  -p 5432:5432 \
  postgres:15
```

### 📊 Database Migrations

After setting up PostgreSQL:

```powershell
# Run migrations
cd "d:\Coding projects\Collab\financial-advisor-tool"
alembic upgrade head
```

---

## 3️⃣ GitHub Secrets (for CI/CD)

### 📍 Location: GitHub Repository → Settings → Secrets and Variables → Actions

**STATUS:** 🔴 **NOT CONFIGURED** - Need to add secrets

### Action Required:

Go to: `https://github.com/YOUR_USERNAME/financial-advisor-tool/settings/secrets/actions`

Add the following secrets:

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| **DOCKER_USERNAME** | 🔴 Docker Hub username | Sign up at https://hub.docker.com |
| **DOCKER_PASSWORD** | 🔴 Docker Hub password/token | Create access token in Docker Hub |
| **AWS_ACCESS_KEY_ID** | 🟡 AWS access key | AWS Console → IAM → Users |
| **AWS_SECRET_ACCESS_KEY** | 🟡 AWS secret key | AWS Console → IAM → Users |
| **SLACK_WEBHOOK_URL** | 🟡 Slack notifications | Create webhook in Slack |
| **DATABASE_URL** | 🟡 Production DB URL | Your production PostgreSQL URL |

### 🐳 Docker Hub Setup:

1. **Create Account:** https://hub.docker.com/signup
2. **Create Access Token:**
   - Go to Account Settings → Security
   - Click "New Access Token"
   - Name it "github-actions"
   - Copy the token (you won't see it again!)
3. **Add to GitHub Secrets:**
   - `DOCKER_USERNAME`: Your Docker Hub username
   - `DOCKER_PASSWORD`: The access token you just created

---

## 4️⃣ AWS Configuration (Optional - for Cloud Deployment)

### 📍 AWS Services Needed

**STATUS:** 🟡 **OPTIONAL** - Only if deploying to AWS

### Services Required:

1. **ECS (Elastic Container Service)** - For running Docker containers
2. **RDS (PostgreSQL)** - For database
3. **S3** - For data storage
4. **CloudWatch** - For monitoring

### Setup Steps:

1. **Create AWS Account:** https://aws.amazon.com/
2. **Create IAM User:**
   ```
   AWS Console → IAM → Users → Add User
   - Name: github-actions-user
   - Access type: Programmatic access
   - Permissions: AmazonECS_FullAccess, AmazonS3FullAccess
   ```
3. **Save credentials** to GitHub Secrets
4. **Update** `.github/workflows/ci-cd.yml` with your cluster/service names

---

## 5️⃣ CI/CD Workflow Configuration

### 📍 Location: `.github/workflows/ci-cd.yml`

**STATUS:** 🟡 **PARTIALLY CONFIGURED** - Needs customization

### Action Required:

Update these lines in `ci-cd.yml`:

```yaml
# Line 180-181: Docker image names
images: |
  YOUR_DOCKER_USERNAME/lensing-api    # ← Change this
  
# Line 210: Streamlit image
tags: YOUR_DOCKER_USERNAME/lensing-webapp:latest  # ← Change this

# Line 248-249: AWS cluster/service names
--cluster lensing-cluster              # ← Change if different
--service lensing-api-service          # ← Change if different

# Line 254: Production URL
curl -f https://api.lensing.example.com/health  # ← Change to your domain
```

---

## 6️⃣ Slack Notifications (Optional)

### 📍 Slack Webhook Setup

**STATUS:** 🟡 **OPTIONAL** - For deployment notifications

### Action Required:

1. **Create Slack Workspace** (if you don't have one)
2. **Create Incoming Webhook:**
   - Go to: https://api.slack.com/apps
   - Create New App → From Scratch
   - Add "Incoming Webhooks" feature
   - Activate and create webhook
   - Copy webhook URL
3. **Add to GitHub Secrets:**
   - Secret name: `SLACK_WEBHOOK_URL`
   - Value: The webhook URL you copied

---

## 7️⃣ Domain Configuration (Optional)

### 📍 Production Domain Setup

**STATUS:** 🟡 **OPTIONAL** - For production deployment

### Action Required:

If deploying to production with custom domain:

1. **Purchase Domain** (e.g., from Namecheap, GoDaddy)
2. **Configure DNS:**
   ```
   A Record:
   api.yourdomain.com → AWS ECS IP
   
   CNAME Record:
   app.yourdomain.com → AWS ECS DNS
   ```
3. **Update URLs** in:
   - `.github/workflows/ci-cd.yml` (line 254)
   - `app/main.py` (API endpoints)
   - Documentation

---

## 8️⃣ Model Files & Data

### 📍 Pre-trained Models

**STATUS:** 🟢 **OK** - Models will be generated during training

### Optional: Download Pre-trained Models

If you have pre-trained models:

```powershell
# Create models directory
mkdir models\checkpoints

# Place your .pth files here:
# - models/checkpoints/best_pinn_model.pth
# - models/checkpoints/transfer_learning_model.pth
```

---

## 9️⃣ Verification Checklist

### ✅ Run This to Verify Setup:

```powershell
# Navigate to project
cd "d:\Coding projects\Collab\financial-advisor-tool"

# Create virtual environment (if not exists)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Check environment variables
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ .env loaded' if os.getenv('DATABASE_URL') else '❌ .env not configured')"

# Test database connection
python -c "from database.database import engine; engine.connect(); print('✅ Database connected')"

# Run tests
pytest tests/ -v

# Start Streamlit
streamlit run app/main.py
```

---

## 🎯 Priority Order

### 🔴 **CRITICAL** (Required for basic functionality):
1. `.env` file with basic settings
2. Secret keys (SECRET_KEY, JWT_SECRET)

### 🟡 **RECOMMENDED** (For full functionality):
3. PostgreSQL database
4. Docker Hub credentials (for CI/CD)

### 🟢 **OPTIONAL** (For production):
5. AWS credentials
6. Slack webhook
7. Custom domain
8. External API keys

---

## 📞 Need Help?

If you encounter issues:

1. **Check logs:** Look in `logs/` directory
2. **Verify configuration:** Run verification script
3. **Test connections:** Use test scripts in `scripts/`
4. **Review docs:** Check `docs/` directory

---

## 🔄 After Configuration

Once everything is set up:

1. ✅ Mark items as complete in this checklist
2. ✅ Run verification script
3. ✅ Commit changes (but NOT the .env file!)
4. ✅ Test locally
5. ✅ Deploy to production

---

**Last Updated:** October 7, 2025  
**Configuration Status:** 🔴 Pending Setup  
**Priority:** HIGH
