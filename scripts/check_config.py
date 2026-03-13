"""
Configuration Checker Script

Checks all required configuration items and highlights what needs your attention.
Run this before deploying to verify everything is properly configured.
"""

import os
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from typing import Dict, List, Tuple

# ANSI color codes
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Color.CYAN}{Color.BOLD}{'='*70}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{text.center(70)}{Color.RESET}")
    print(f"{Color.CYAN}{Color.BOLD}{'='*70}{Color.RESET}\n")


def print_section(text: str):
    """Print section header"""
    print(f"\n{Color.BLUE}{Color.BOLD}{'─'*70}{Color.RESET}")
    print(f"{Color.BLUE}{Color.BOLD}📋 {text}{Color.RESET}")
    print(f"{Color.BLUE}{Color.BOLD}{'─'*70}{Color.RESET}\n")


def check_status(condition: bool, success_msg: str, failure_msg: str) -> bool:
    """Print status with color"""
    if condition:
        print(f"  {Color.GREEN}✅ {success_msg}{Color.RESET}")
        return True
    else:
        print(f"  {Color.RED}❌ {failure_msg}{Color.RESET}")
        return False


def check_warning(condition: bool, success_msg: str, warning_msg: str) -> bool:
    """Print warning status"""
    if condition:
        print(f"  {Color.GREEN}✅ {success_msg}{Color.RESET}")
        return True
    else:
        print(f"  {Color.YELLOW}⚠️  {warning_msg}{Color.RESET}")
        return False


def check_env_file() -> Dict[str, bool]:
    """Check .env file configuration"""
    print_section("1. Environment Variables (.env)")
    
    results = {}
    env_path = Path('.env')
    
    # Check if .env exists
    if not env_path.exists():
        print(f"  {Color.RED}❌ .env file NOT FOUND{Color.RESET}")
        print(f"  {Color.YELLOW}   ACTION REQUIRED: Create .env file in project root{Color.RESET}")
        print(f"  {Color.YELLOW}   See CONFIG_SETUP.md for template{Color.RESET}")
        return {'env_file_exists': False}
    
    print(f"  {Color.GREEN}✅ .env file found{Color.RESET}")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print(f"  {Color.GREEN}✅ .env file loaded successfully{Color.RESET}")
    except ImportError:
        print(f"  {Color.YELLOW}⚠️  python-dotenv not installed (optional){Color.RESET}")
    
    # Check critical variables
    critical_vars = {
        'SECRET_KEY': 'Secret key for encryption',
        'JWT_SECRET': 'JWT token secret',
    }
    
    optional_vars = {
        'DATABASE_URL': 'PostgreSQL database connection',
        'REDIS_URL': 'Redis cache connection',
        'AWS_ACCESS_KEY_ID': 'AWS credentials',
        'SLACK_WEBHOOK_URL': 'Slack notifications',
    }
    
    print(f"\n  {Color.BOLD}Critical Variables:{Color.RESET}")
    for var, desc in critical_vars.items():
        value = os.getenv(var)
        if value and value != f'your-{var.lower()}-here' and len(value) > 10:
            results[var] = check_status(True, f"{var}: Configured", "")
        else:
            results[var] = check_status(False, "", 
                f"{var}: NOT CONFIGURED - {desc}")
            print(f"    {Color.YELLOW}→ Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\"{Color.RESET}")
    
    print(f"\n  {Color.BOLD}Optional Variables:{Color.RESET}")
    for var, desc in optional_vars.items():
        value = os.getenv(var)
        results[var] = check_warning(
            bool(value) and value != '',
            f"{var}: Configured",
            f"{var}: Not set - {desc} (optional)"
        )
    
    return results


def check_database() -> bool:
    """Check database connection"""
    print_section("2. Database Configuration")
    
    try:
        from database.database import engine
        print(f"  {Color.GREEN}✅ Database module imported{Color.RESET}")
        
        # Try to connect
        try:
            with engine.connect() as conn:
                print(f"  {Color.GREEN}✅ Database connection successful{Color.RESET}")
                return True
        except Exception as e:
            print(f"  {Color.RED}❌ Cannot connect to database{Color.RESET}")
            print(f"    {Color.YELLOW}Error: {str(e)}{Color.RESET}")
            print(f"    {Color.YELLOW}ACTION REQUIRED: Configure PostgreSQL{Color.RESET}")
            print(f"    {Color.YELLOW}See CONFIG_SETUP.md Section 2{Color.RESET}")
            return False
    except ImportError:
        print(f"  {Color.YELLOW}⚠️  Database module not found (optional for basic use){Color.RESET}")
        return False


def check_docker() -> bool:
    """Check Docker configuration"""
    print_section("3. Docker Configuration")
    
    import subprocess
    
    # Check if Docker is installed
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  {Color.GREEN}✅ Docker installed: {result.stdout.strip()}{Color.RESET}")
        else:
            print(f"  {Color.YELLOW}⚠️  Docker not found (optional){Color.RESET}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"  {Color.YELLOW}⚠️  Docker not installed (optional for local dev){Color.RESET}")
        return False
    
    # Check Docker Hub credentials in environment
    docker_user = os.getenv('DOCKER_USERNAME')
    docker_pass = os.getenv('DOCKER_PASSWORD')
    
    if docker_user and docker_pass:
        print(f"  {Color.GREEN}✅ Docker Hub credentials configured{Color.RESET}")
        return True
    else:
        print(f"  {Color.YELLOW}⚠️  Docker Hub credentials not set{Color.RESET}")
        print(f"    {Color.YELLOW}Required for: CI/CD pipeline{Color.RESET}")
        print(f"    {Color.YELLOW}See CONFIG_SETUP.md Section 3{Color.RESET}")
        return False


def check_github_secrets() -> bool:
    """Check GitHub configuration"""
    print_section("4. GitHub CI/CD Configuration")
    
    # Check if .github/workflows exists
    workflows_path = Path('.github/workflows')
    if workflows_path.exists():
        print(f"  {Color.GREEN}✅ GitHub workflows directory found{Color.RESET}")
        
        # List workflow files
        workflow_files = list(workflows_path.glob('*.yml'))
        if workflow_files:
            print(f"  {Color.GREEN}✅ Found {len(workflow_files)} workflow(s):{Color.RESET}")
            for wf in workflow_files:
                print(f"    • {wf.name}")
        
        print(f"\n  {Color.YELLOW}⚠️  GitHub Secrets Configuration:{Color.RESET}")
        print(f"    {Color.YELLOW}The following secrets need to be set in GitHub:{Color.RESET}")
        secrets = [
            ('DOCKER_USERNAME', 'Required for CI/CD'),
            ('DOCKER_PASSWORD', 'Required for CI/CD'),
            ('AWS_ACCESS_KEY_ID', 'Optional for AWS deployment'),
            ('AWS_SECRET_ACCESS_KEY', 'Optional for AWS deployment'),
            ('SLACK_WEBHOOK_URL', 'Optional for notifications'),
        ]
        for secret, desc in secrets:
            print(f"    • {Color.CYAN}{secret}{Color.RESET} - {desc}")
        
        print(f"\n    {Color.YELLOW}→ Go to: GitHub Repo → Settings → Secrets and Variables → Actions{Color.RESET}")
        print(f"    {Color.YELLOW}→ See CONFIG_SETUP.md Section 3 for details{Color.RESET}")
        
        return True
    else:
        print(f"  {Color.YELLOW}⚠️  GitHub workflows not found{Color.RESET}")
        return False


def check_modules() -> bool:
    """Check if all required modules work"""
    print_section("5. Python Modules & Dependencies")
    
    required_modules = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('fastapi', 'FastAPI'),
        ('matplotlib', 'Matplotlib'),
    ]
    
    optional_modules = [
        ('cupy', 'CuPy (GPU acceleration)'),
        ('redis', 'Redis (caching)'),
        ('psycopg2', 'PostgreSQL adapter'),
        ('dotenv', 'Environment variables'),
    ]
    
    all_ok = True
    
    print(f"  {Color.BOLD}Required Modules:{Color.RESET}")
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"  {Color.GREEN}✅ {name}{Color.RESET}")
        except ImportError:
            print(f"  {Color.RED}❌ {name} - NOT INSTALLED{Color.RESET}")
            all_ok = False
    
    print(f"\n  {Color.BOLD}Optional Modules:{Color.RESET}")
    for module, name in optional_modules:
        try:
            __import__(module)
            print(f"  {Color.GREEN}✅ {name}{Color.RESET}")
        except ImportError:
            print(f"  {Color.YELLOW}⚠️  {name} - Not installed (optional){Color.RESET}")
    
    return all_ok


def check_project_modules() -> bool:
    """Check project-specific modules"""
    print_section("6. Project Modules")
    
    modules_to_check = [
        ('src.lens_models', 'Lens Models'),
        ('src.ml.pinn', 'Physics-Informed Neural Networks'),
        ('src.validation', 'Scientific Validation'),
        ('src.ml.uncertainty', 'Bayesian Uncertainty Quantification'),
        ('api.main', 'FastAPI Application'),
    ]
    
    all_ok = True
    for module, name in modules_to_check:
        try:
            __import__(module)
            print(f"  {Color.GREEN}✅ {name}{Color.RESET}")
        except ImportError as e:
            print(f"  {Color.RED}❌ {name} - IMPORT ERROR{Color.RESET}")
            print(f"    {Color.YELLOW}Error: {str(e)}{Color.RESET}")
            all_ok = False
    
    return all_ok


def check_tests() -> bool:
    """Check if tests run"""
    print_section("7. Test Suite")
    
    test_scripts = [
        ('scripts/test_validator.py', 'Validation Tests'),
        ('scripts/test_bayesian_uq.py', 'Bayesian UQ Tests'),
    ]
    
    print(f"  {Color.YELLOW}ℹ️  To run tests manually:{Color.RESET}")
    for script, name in test_scripts:
        if Path(script).exists():
            print(f"    {Color.CYAN}python {script}{Color.RESET} - {name}")
        else:
            print(f"    {Color.RED}❌ {script} not found{Color.RESET}")
    
    print(f"\n  {Color.YELLOW}ℹ️  Or run all tests with: pytest tests/ -v{Color.RESET}")
    return True


def print_summary(results: Dict[str, bool]):
    """Print configuration summary"""
    print_header("CONFIGURATION SUMMARY")
    
    total = len(results)
    configured = sum(results.values())
    percentage = (configured / total * 100) if total > 0 else 0
    
    # Status
    if percentage == 100:
        status_color = Color.GREEN
        status = "✅ FULLY CONFIGURED"
    elif percentage >= 50:
        status_color = Color.YELLOW
        status = "⚠️  PARTIALLY CONFIGURED"
    else:
        status_color = Color.RED
        status = "❌ NEEDS CONFIGURATION"
    
    print(f"{status_color}{Color.BOLD}{status}{Color.RESET}")
    print(f"\nConfiguration Status: {status_color}{configured}/{total} items configured ({percentage:.1f}%){Color.RESET}")
    
    # What needs attention
    unconfigured = [k for k, v in results.items() if not v]
    if unconfigured:
        print(f"\n{Color.RED}{Color.BOLD}⚠️  ACTION REQUIRED:{Color.RESET}")
        print(f"{Color.YELLOW}The following items need your configuration:{Color.RESET}\n")
        for item in unconfigured:
            print(f"  • {Color.YELLOW}{item}{Color.RESET}")
        print(f"\n{Color.CYAN}→ See CONFIG_SETUP.md for detailed setup instructions{Color.RESET}")
    else:
        print(f"\n{Color.GREEN}🎉 All configuration items are set up!{Color.RESET}")
    
    # Next steps
    print(f"\n{Color.BOLD}Next Steps:{Color.RESET}")
    if percentage < 100:
        print(f"  1. {Color.YELLOW}Review CONFIG_SETUP.md{Color.RESET}")
        print(f"  2. {Color.YELLOW}Configure missing items{Color.RESET}")
        print(f"  3. {Color.YELLOW}Run this script again to verify{Color.RESET}")
    else:
        print(f"  1. {Color.GREEN}Run tests: pytest tests/ -v{Color.RESET}")
        print(f"  2. {Color.GREEN}Start API: uvicorn api.main:app --reload{Color.RESET}")
        print(f"  3. {Color.GREEN}Deploy to production (if ready){Color.RESET}")


def main():
    """Main configuration checker"""
    print_header("🔧 GRAVITATIONAL LENSING TOOLKIT")
    print_header("Configuration Checker")
    
    print(f"{Color.CYAN}This script checks your configuration and highlights items needing attention.{Color.RESET}")
    print(f"{Color.CYAN}See CONFIG_SETUP.md for detailed setup instructions.{Color.RESET}")
    
    results = {}
    
    # Run all checks
    env_results = check_env_file()
    results.update(env_results)
    
    results['database'] = check_database()
    results['docker'] = check_docker()
    results['github'] = check_github_secrets()
    results['python_modules'] = check_modules()
    results['project_modules'] = check_project_modules()
    results['tests'] = check_tests()
    
    # Print summary
    print_summary(results)
    
    print(f"\n{Color.CYAN}{'='*70}{Color.RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Color.YELLOW}Configuration check interrupted by user.{Color.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Color.RED}Error running configuration check: {e}{Color.RESET}")
        sys.exit(1)
