# Upload to GitHub in batches
# Repository: https://github.com/nalin1304/Gravitational-Lensing-algorithm

Write-Host "🚀 Uploading to GitHub in batches..." -ForegroundColor Cyan

# Check if remote exists, if not add it
$remoteExists = git remote | Select-String "origin"
if (-not $remoteExists) {
    Write-Host "Adding remote origin..." -ForegroundColor Yellow
    git remote add origin https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
}

# Rename branch to main
Write-Host "Renaming branch to main..." -ForegroundColor Yellow
git branch -M main

# Batch 1: Core application files
Write-Host "`n📦 Batch 1: Core application and source code" -ForegroundColor Green
git add app/ src/ tests/
git commit -m "Add: Core application, source code, and tests"
git push -u origin main

# Batch 2: Documentation
Write-Host "`n📦 Batch 2: Documentation" -ForegroundColor Green
git add docs/ *.md
git commit -m "Add: Documentation and markdown files"
git push origin main

# Batch 3: Configuration and scripts
Write-Host "`n📦 Batch 3: Configuration and scripts" -ForegroundColor Green
git add scripts/ api/ database/ benchmarks/ monitoring/
git commit -m "Add: Scripts, API, database, and monitoring"
git push origin main

# Batch 4: Docker and CI/CD
Write-Host "`n📦 Batch 4: Docker and CI/CD" -ForegroundColor Green
git add .github/ .gitignore Dockerfile* docker-compose.yml .env.example
git commit -m "Add: Docker configuration and CI/CD workflows"
git push origin main

# Batch 5: Notebooks and results
Write-Host "`n📦 Batch 5: Notebooks and results" -ForegroundColor Green
git add notebooks/ results/
git commit -m "Add: Jupyter notebooks and results"
git push origin main

# Batch 6: Remaining files
Write-Host "`n📦 Batch 6: Remaining configuration files" -ForegroundColor Green
git add .
git commit -m "Add: Remaining configuration and utility files"
git push origin main

Write-Host "`n✅ Upload complete!" -ForegroundColor Green
Write-Host "🌐 View at: https://github.com/nalin1304/Gravitational-Lensing-algorithm" -ForegroundColor Cyan
