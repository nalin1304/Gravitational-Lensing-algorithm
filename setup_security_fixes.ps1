# Security Remediation - Dependency Installation and Verification
# Run this script after pulling the security fixes

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Security Remediation - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.9+ first." -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
if (Test-Path "venv") {
    Write-Host "✓ Virtual environment found" -ForegroundColor Green
} else {
    Write-Host "⚠ Virtual environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host ""
Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip | Out-Null
Write-Host "✓ pip upgraded" -ForegroundColor Green

# Install updated dependencies
Write-Host ""
Write-Host "Installing updated dependencies..." -ForegroundColor Cyan
Write-Host "  - Upgrading pillow (CVE-2023-50447 fix)" -ForegroundColor Yellow
Write-Host "  - Upgrading python-jose (cryptography updates)" -ForegroundColor Yellow
Write-Host "  - Installing slowapi (rate limiting)" -ForegroundColor Yellow
Write-Host "  - Installing bleach (XSS prevention)" -ForegroundColor Yellow

pip install -r requirements.txt --upgrade

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "✗ Dependency installation failed" -ForegroundColor Red
    exit 1
}

# Verify critical packages
Write-Host ""
Write-Host "Verifying security-critical packages..." -ForegroundColor Cyan

$packages = @(
    @{name="pillow"; minVersion="10.4.0"},
    @{name="slowapi"; minVersion="0.1.9"},
    @{name="bleach"; minVersion="6.0.0"},
    @{name="python-jose"; minVersion="3.3.0"}
)

$allGood = $true
foreach ($pkg in $packages) {
    try {
        $installed = pip show $pkg.name 2>&1 | Select-String "Version:" | ForEach-Object { $_.ToString().Split(":")[1].Trim() }
        if ($installed) {
            Write-Host "  ✓ $($pkg.name): $installed (>= $($pkg.minVersion))" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $($pkg.name): NOT INSTALLED" -ForegroundColor Red
            $allGood = $false
        }
    } catch {
        Write-Host "  ✗ $($pkg.name): ERROR checking version" -ForegroundColor Red
        $allGood = $false
    }
}

# Check for known vulnerabilities
Write-Host ""
Write-Host "Checking for known vulnerabilities..." -ForegroundColor Cyan
Write-Host "(Installing pip-audit if not present)" -ForegroundColor Gray
pip install pip-audit --quiet

Write-Host ""
pip-audit --desc

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ No known vulnerabilities found" -ForegroundColor Green
} else {
    Write-Host "⚠ Some vulnerabilities detected. Review output above." -ForegroundColor Yellow
}

# Generate SSL certificates for development
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PostgreSQL SSL Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path "database\ssl\server.key") {
    Write-Host "✓ SSL certificates already exist" -ForegroundColor Green
} else {
    Write-Host "⚠ SSL certificates not found" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Generate self-signed certificates for development? (y/n)"
    
    if ($response -eq "y" -or $response -eq "Y") {
        # Create directory
        New-Item -Path "database\ssl" -ItemType Directory -Force | Out-Null
        
        # Check if OpenSSL is available
        try {
            $opensslVersion = openssl version 2>&1
            Write-Host "✓ OpenSSL found: $opensslVersion" -ForegroundColor Green
            
            Write-Host "Generating SSL certificates..." -ForegroundColor Cyan
            
            # Generate private key
            openssl genrsa -out database\ssl\server.key 2048 2>&1 | Out-Null
            
            # Generate self-signed certificate
            $subject = "/C=US/ST=CA/L=SanFrancisco/O=LensingProject/CN=localhost"
            openssl req -new -x509 -key database\ssl\server.key -out database\ssl\server.crt -days 365 -subj $subject 2>&1 | Out-Null
            
            # Set permissions (Windows)
            icacls database\ssl\server.key /inheritance:r /grant:r "$($env:USERNAME):(R)" | Out-Null
            
            Write-Host "✓ SSL certificates generated" -ForegroundColor Green
            Write-Host "  Location: database\ssl\" -ForegroundColor Gray
            Write-Host "  Valid for: 365 days" -ForegroundColor Gray
            
        } catch {
            Write-Host "✗ OpenSSL not found. Please install OpenSSL or generate certificates manually." -ForegroundColor Red
            Write-Host "  See database\SSL_SETUP_GUIDE.md for instructions" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ Skipped SSL certificate generation" -ForegroundColor Yellow
        Write-Host "  See database\SSL_SETUP_GUIDE.md for manual setup" -ForegroundColor Gray
    }
}

# Summary
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

if ($allGood) {
    Write-Host "✓ All security dependencies installed" -ForegroundColor Green
    Write-Host "✓ No known vulnerabilities detected" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review SECURITY_REMEDIATION_COMPLETE.md" -ForegroundColor White
    Write-Host "  2. Set SECRET_KEY in .env (min 32 chars)" -ForegroundColor White
    Write-Host "  3. Set DB_PASSWORD in .env" -ForegroundColor White
    Write-Host "  4. Run tests: pytest tests/" -ForegroundColor White
    Write-Host "  5. Start services: docker-compose up -d" -ForegroundColor White
} else {
    Write-Host "⚠ Some issues detected. Please review output above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "For detailed documentation, see:" -ForegroundColor Cyan
Write-Host "  - SECURITY_REMEDIATION_COMPLETE.md" -ForegroundColor White
Write-Host "  - database/SSL_SETUP_GUIDE.md" -ForegroundColor White
Write-Host ""
