"""
Comprehensive Code Audit Script
Checks for all potential issues in the Streamlit app
"""

import re
from pathlib import Path

def audit_file(filepath):
    """Audit a Python file for potential issues"""
    issues = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        # Check for unprotected attribute access on session_state
        if re.search(r'st\.session_state\[[\'"][^\'"]+[\'"]\]\.\w+\(', line):
            if 'if' not in lines[max(0, i-3):i] and 'hasattr' not in lines[max(0, i-3):i]:
                issues.append({
                    'line': i,
                    'type': 'CRITICAL',
                    'issue': 'Unprotected session_state attribute access',
                    'code': line.strip()
                })
        
        # Check for exposed tracebacks NOT in expanders
        if 'st.code(traceback.format_exc())' in line:
            # Check if wrapped in expander (look at previous lines)
            context = '\n'.join(lines[max(0, i-5):i])
            if 'with st.expander' not in context:
                issues.append({
                    'line': i,
                    'type': 'HIGH',
                    'issue': 'Exposed traceback (not in expander)',
                    'code': line.strip()
                })
        
        # Check for hard returns that block features
        if 'return' in line and ('not MODULES_AVAILABLE' in lines[max(0, i-5):i] or 
                                  'not PHASE15_AVAILABLE' in lines[max(0, i-5):i] or
                                  'not ASTROPY_AVAILABLE' in lines[max(0, i-5):i]):
            # Check if it's a hard block
            context = '\n'.join(lines[max(0, i-10):i+1])
            if 'st.error' in context or 'st.warning' in context:
                issues.append({
                    'line': i,
                    'type': 'MEDIUM',
                    'issue': 'Hard feature block (consider graceful degradation)',
                    'code': line.strip()
                })
        
        # Check for None access without validation
        if re.search(r'(\w+)\.(max|min|mean|shape|dtype)', line):
            # Check if there's validation nearby
            context = '\n'.join(lines[max(0, i-5):i])
            if 'if' not in context and 'is not None' not in context and 'hasattr' not in context:
                if 'def ' not in context:  # Skip function definitions
                    issues.append({
                        'line': i,
                        'type': 'MEDIUM',
                        'issue': 'Potential None access without check',
                        'code': line.strip()
                    })
    
    return issues

def main():
    """Run comprehensive audit"""
    main_py = Path('app/main.py')
    
    print("=" * 80)
    print("COMPREHENSIVE CODE AUDIT")
    print("=" * 80)
    print()
    
    if not main_py.exists():
        print(f"❌ File not found: {main_py}")
        return
    
    issues = audit_file(main_py)
    
    # Categorize issues
    critical = [i for i in issues if i['type'] == 'CRITICAL']
    high = [i for i in issues if i['type'] == 'HIGH']
    medium = [i for i in issues if i['type'] == 'MEDIUM']
    
    print(f"📊 AUDIT SUMMARY")
    print(f"   Total Issues: {len(issues)}")
    print(f"   🔴 Critical: {len(critical)}")
    print(f"   🟠 High: {len(high)}")
    print(f"   🟡 Medium: {len(medium)}")
    print()
    
    if critical:
        print("🔴 CRITICAL ISSUES (Must Fix):")
        print("-" * 80)
        for issue in critical:
            print(f"   Line {issue['line']}: {issue['issue']}")
            print(f"   Code: {issue['code']}")
            print()
    
    if high:
        print("🟠 HIGH PRIORITY ISSUES:")
        print("-" * 80)
        for issue in high:
            print(f"   Line {issue['line']}: {issue['issue']}")
            print(f"   Code: {issue['code']}")
            print()
    
    if medium:
        print(f"🟡 MEDIUM PRIORITY ISSUES: {len(medium)} found")
        print("   (Run with --verbose to see all)")
        print()
    
    # Generate fix recommendations
    print("=" * 80)
    print("📝 RECOMMENDED FIXES")
    print("=" * 80)
    
    if critical:
        print("\n1. Fix session_state access:")
        print("   Replace: st.session_state['key'].method()")
        print("   With: data = st.session_state.get('key'); if data is not None: data.method()")
    
    if high:
        print("\n2. Wrap exposed tracebacks:")
        print("   Replace: st.code(traceback.format_exc())")
        print("   With: with st.expander('🔍 Details'): st.code(traceback.format_exc())")
    
    if medium:
        print("\n3. Add None checks before attribute access")
    
    print()
    print("=" * 80)
    
    if not issues:
        print("✅ NO ISSUES FOUND! Code looks good!")
    else:
        print(f"⚠️ Found {len(issues)} issues that need attention")
    
    return issues

if __name__ == "__main__":
    issues = main()
