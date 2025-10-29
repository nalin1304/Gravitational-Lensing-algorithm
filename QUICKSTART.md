# 🚀 Quick Start Guide

## ⚠️ BEFORE YOU BEGIN - Configuration Required

**Several items need your manual configuration before the system is fully functional.**

### ✅ Run Configuration Checker First:

```powershell
cd "d:\Coding projects\Collab\financial-advisor-tool"
python scripts/check_config.py
```

This will show you exactly what needs to be configured with color-coded status.

---

## 📋 Essential Setup (5 Minutes)

### 1️⃣ Create .env File

```powershell
# Copy template
cp .env.example .env
```

### 2️⃣ Generate Secret Keys

```powershell
# Generate SECRET_KEY
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"

# Generate JWT_SECRET  
python -c "import secrets; print('JWT_SECRET=' + secrets.token_urlsafe(32))"
```

**Copy these values into your `.env` file**

### 3️⃣ Basic .env Configuration

Edit `.env` and set at minimum:

```env
# 🔴 REQUIRED
SECRET_KEY=<paste_generated_key_here>
JWT_SECRET=<paste_generated_key_here>
ENVIRONMENT=development
DEBUG=True
```

### 4️⃣ Verify Setup

```powershell
python scripts/check_config.py
```

---

## 🎯 Run the Application

### Start Streamlit UI:

```powershell
cd "d:\Coding projects\Collab\financial-advisor-tool"
streamlit run app/main.py
```

**Access at:** http://localhost:8501

### Run Tests:

```powershell
# All Phase 15 tests
python scripts/test_validator.py
python scripts/test_bayesian_uq.py

# Full test suite
pytest tests/ -v
```

---

## 📖 Detailed Configuration

For complete setup including:
- 🟡 **Database** (PostgreSQL)
- 🟡 **Docker Hub** (for CI/CD)
- 🟡 **AWS** (cloud deployment)
- 🟡 **Slack** (notifications)

**See:** [`CONFIG_SETUP.md`](CONFIG_SETUP.md) - Comprehensive configuration guide

---

## 🔴 Items Requiring YOUR Action

| Priority | Item | Where to Configure | Time Required |
|----------|------|-------------------|---------------|
| 🔴 **HIGH** | Secret Keys | `.env` file | 2 min |
| 🔴 **HIGH** | Environment Type | `.env` → `ENVIRONMENT` | 1 min |
| 🟡 **MEDIUM** | Database | See CONFIG_SETUP.md § 2 | 10 min |
| 🟡 **MEDIUM** | Docker Hub | See CONFIG_SETUP.md § 3 | 5 min |
| 🟡 **MEDIUM** | GitHub Secrets | GitHub → Settings → Secrets | 5 min |
| 🟢 **LOW** | AWS Credentials | See CONFIG_SETUP.md § 4 | 15 min |
| 🟢 **LOW** | Slack Webhook | See CONFIG_SETUP.md § 6 | 5 min |

---

## 🎨 Features

### ✅ Fully Working (No Configuration Needed):
- Scientific validation
- Bayesian uncertainty quantification  
- Synthetic data generation
- PINN training & inference
- Interactive visualizations
- All 17 Phase 15 tests

### 🟡 Optional (Needs Configuration):
- PostgreSQL database integration
- CI/CD pipeline (Docker Hub + GitHub Actions)
- Cloud deployment (AWS)
- Slack notifications
- Production monitoring

---

## 📁 Project Structure

```
financial-advisor-tool/
├── 📝 CONFIG_SETUP.md          ← 🔴 READ THIS - Complete setup guide
├── 📝 .env.example             ← 🔴 Copy to .env and configure
├── 🔧 scripts/
│   ├── check_config.py         ← 🔴 RUN THIS to verify setup
│   ├── test_validator.py       ← Test validation
│   └── test_bayesian_uq.py     ← Test UQ
├── 📱 app/
│   └── main.py                 ← Streamlit application
├── 🧬 src/
│   ├── lens_models/            ← Gravitational lensing
│   ├── ml/                     ← Machine learning
│   └── validation/             ← Scientific validation
└── 🧪 tests/                   ← Test suite
```

---

## 🚦 Status Check

Run this to see configuration status:

```powershell
python scripts/check_config.py
```

**Output shows:**
- ✅ Green = Configured correctly
- ⚠️ Yellow = Optional (recommended)
- ❌ Red = Required (needs attention)

---

## 💡 Common Questions

### Q: Do I need PostgreSQL?
**A:** No, for basic functionality. Yes, for full database features and production.

### Q: Do I need Docker Hub credentials?
**A:** Only if you want to use the CI/CD pipeline (GitHub Actions).

### Q: Do I need AWS credentials?
**A:** Only if deploying to AWS cloud. Works fine locally without it.

### Q: What's the bare minimum to run?
**A:** Just create `.env` with SECRET_KEY and JWT_SECRET. Everything else is optional.

---

## 🆘 Troubleshooting

### Configuration Checker Shows Errors?
```powershell
python scripts/check_config.py
```
Follow the colored indicators and refer to CONFIG_SETUP.md

### Tests Failing?
```powershell
# Clear cache and retry
rm -r .streamlit/cache
python scripts/test_validator.py
```

### Import Errors?
```powershell
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 📞 Next Steps

1. ✅ **Run:** `python scripts/check_config.py`
2. ✅ **Create:** `.env` file with secret keys
3. ✅ **Test:** `python scripts/test_validator.py`
4. ✅ **Launch:** `streamlit run app/main.py`
5. 📖 **Review:** `CONFIG_SETUP.md` for optional features

---

**Status:** Phase 15 Complete - All Core Features Working  
**Last Updated:** October 7, 2025  
**Configuration Required:** Yes (see CONFIG_SETUP.md)
