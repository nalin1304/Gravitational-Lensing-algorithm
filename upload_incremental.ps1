# Simple GitHub Upload - Incremental Approach
# This script uploads files in small batches to avoid GitHub's file limits

Write-Host "Starting incremental upload to GitHub..." -ForegroundColor Cyan
Write-Host "Repository: https://github.com/nalin1304/Gravitational-Lensing-algorithm" -ForegroundColor Yellow
Write-Host ""

# Step 1: Add remote if not exists
Write-Host "Step 1: Configuring remote..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin https://github.com/nalin1304/Gravitational-Lensing-algorithm.git
git branch -M main

# Step 2: Create initial commit with essential files only
Write-Host "`nStep 2: Creating initial commit (essential files)..." -ForegroundColor Green
git reset
git add README.md LICENSE .gitignore requirements.txt
git commit -m "Initial commit: Essential files"

# Step 3: Push initial commit
Write-Host "`nStep 3: Pushing initial commit..." -ForegroundColor Green
git push -u origin main --force

Write-Host "`nWaiting 3 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Step 4: Add app directory
Write-Host "`nStep 4: Adding app directory..." -ForegroundColor Green
git add app/
git commit -m "Add: Streamlit application"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 5: Add src directory
Write-Host "`nStep 5: Adding src directory..." -ForegroundColor Green
git add src/
git commit -m "Add: Core source code modules"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 6: Add tests
Write-Host "`nStep 6: Adding tests..." -ForegroundColor Green
git add tests/ test_*.py
git commit -m "Add: Test suite"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 7: Add documentation
Write-Host "`nStep 7: Adding documentation..." -ForegroundColor Green
git add docs/ *.md
git commit -m "Add: Documentation"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 8: Add configuration
Write-Host "`nStep 8: Adding configuration..." -ForegroundColor Green
git add scripts/ api/ database/ benchmarks/
git commit -m "Add: Scripts and configuration"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 9: Add CI/CD
Write-Host "`nStep 9: Adding CI/CD..." -ForegroundColor Green
git add .github/ .env.example
git commit -m "Add: GitHub workflows and CI/CD"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 10: Add Docker
Write-Host "`nStep 10: Adding Docker..." -ForegroundColor Green
git add Dockerfile* docker-compose.yml
git commit -m "Add: Docker configuration"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 11: Add notebooks
Write-Host "`nStep 11: Adding notebooks..." -ForegroundColor Green
git add notebooks/
git commit -m "Add: Jupyter notebooks"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 12: Add results
Write-Host "`nStep 12: Adding results..." -ForegroundColor Green
git add results/
git commit -m "Add: Results and visualizations"
git push origin main

Write-Host "`n⏳ Waiting 2 seconds..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Step 13: Add remaining files
Write-Host "`nStep 13: Adding remaining files..." -ForegroundColor Green
git add .
git commit -m "Add: Remaining configuration files"
git push origin main

Write-Host "`n===== UPLOAD COMPLETE! =====" -ForegroundColor Green
Write-Host "Repository: https://github.com/nalin1304/Gravitational-Lensing-algorithm" -ForegroundColor Yellow
Write-Host "Check your repository now!" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Cyan
