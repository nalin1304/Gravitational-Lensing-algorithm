# Simple GitHub Upload Script
# Uploads files in batches to avoid GitHub limits

Write-Host "Starting upload to GitHub..." -ForegroundColor Cyan
Write-Host "Repository: https://github.com/nalin1304/Gravitational-Lensing-algorithm"
Write-Host ""

# Configure remote
Write-Host "Configuring remote..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
git branch -M main

# Reset and start fresh
Write-Host "Resetting repository..." -ForegroundColor Yellow
git reset

# Batch 1: Essential files
Write-Host "`nBatch 1: Essential files..." -ForegroundColor Green
git add README.md LICENSE .gitignore requirements.txt
git commit -m "Initial commit: Essential files"
git push -u origin main --force
Start-Sleep -Seconds 2

# Batch 2: App
Write-Host "Batch 2: App directory..." -ForegroundColor Green
git add app/
git commit -m "Add: Streamlit application"
git push origin main
Start-Sleep -Seconds 2

# Batch 3: Source code
Write-Host "Batch 3: Source code..." -ForegroundColor Green
git add src/
git commit -m "Add: Core source modules"
git push origin main
Start-Sleep -Seconds 2

# Batch 4: Tests
Write-Host "Batch 4: Tests..." -ForegroundColor Green
git add tests/ test_*.py
git commit -m "Add: Test suite"
git push origin main
Start-Sleep -Seconds 2

# Batch 5: Documentation
Write-Host "Batch 5: Documentation..." -ForegroundColor Green
git add docs/
git commit -m "Add: Documentation (part 1)"
git push origin main
Start-Sleep -Seconds 2

# Batch 6: More docs
Write-Host "Batch 6: More documentation..." -ForegroundColor Green
git add *.md
git commit -m "Add: Markdown documentation"
git push origin main
Start-Sleep -Seconds 2

# Batch 7: Scripts
Write-Host "Batch 7: Scripts..." -ForegroundColor Green
git add scripts/ api/ database/
git commit -m "Add: Scripts and database"
git push origin main
Start-Sleep -Seconds 2

# Batch 8: Benchmarks
Write-Host "Batch 8: Benchmarks..." -ForegroundColor Green
git add benchmarks/ monitoring/
git commit -m "Add: Benchmarks and monitoring"
git push origin main
Start-Sleep -Seconds 2

# Batch 9: GitHub workflows
Write-Host "Batch 9: GitHub workflows..." -ForegroundColor Green
git add .github/ .env.example
git commit -m "Add: CI/CD workflows"
git push origin main
Start-Sleep -Seconds 2

# Batch 10: Docker
Write-Host "Batch 10: Docker..." -ForegroundColor Green
git add Dockerfile* docker-compose.yml alembic.ini
git commit -m "Add: Docker configuration"
git push origin main
Start-Sleep -Seconds 2

# Batch 11: Notebooks
Write-Host "Batch 11: Notebooks..." -ForegroundColor Green
git add notebooks/
git commit -m "Add: Jupyter notebooks"
git push origin main
Start-Sleep -Seconds 2

# Batch 12: Results
Write-Host "Batch 12: Results..." -ForegroundColor Green
git add results/
git commit -m "Add: Results and visualizations"
git push origin main
Start-Sleep -Seconds 2

# Batch 13: Migrations
Write-Host "Batch 13: Migrations..." -ForegroundColor Green
git add migrations/
git commit -m "Add: Database migrations"
git push origin main
Start-Sleep -Seconds 2

# Batch 14: Everything else
Write-Host "Batch 14: Remaining files..." -ForegroundColor Green
git add .
git commit -m "Add: Remaining files"
git push origin main

Write-Host "`n===== UPLOAD COMPLETE =====" -ForegroundColor Green
Write-Host "View at: https://github.com/nalin1304/Gravitational-Lensing-algorithm" -ForegroundColor Yellow
