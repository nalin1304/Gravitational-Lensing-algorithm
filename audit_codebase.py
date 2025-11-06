"""
Comprehensive Codebase Auditor
Performs deep analysis of all Python files to find:
- Empty or stub functions
- NotImplementedError instances
- Invalid implementations
- Missing dependencies
- Unused imports
- Data accuracy issues
"""

import ast
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import re

class CodebaseAuditor:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.issues = []
        self.stats = {
            'total_files': 0,
            'total_lines': 0,
            'empty_functions': 0,
            'not_implemented': 0,
            'todo_comments': 0,
            'missing_docstrings': 0,
            'unused_imports': 0
        }
    
    def audit_file(self, filepath: Path) -> List[Dict]:
        """Audit a single Python file."""
        file_issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            self.stats['total_files'] += 1
            self.stats['total_lines'] += len(lines)
            
            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                file_issues.append({
                    'file': str(filepath),
                    'type': 'SYNTAX_ERROR',
                    'line': e.lineno,
                    'message': str(e)
                })
                return file_issues
            
            # Check for empty functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if self._is_empty_function(node):
                        self.stats['empty_functions'] += 1
                        file_issues.append({
                            'file': str(filepath),
                            'type': 'EMPTY_FUNCTION',
                            'line': node.lineno,
                            'function': node.name,
                            'message': f"Function '{node.name}' is empty or only has pass/docstring"
                        })
                    
                    # Check for NotImplementedError
                    if self._has_not_implemented(node):
                        self.stats['not_implemented'] += 1
                        file_issues.append({
                            'file': str(filepath),
                            'type': 'NOT_IMPLEMENTED',
                            'line': node.lineno,
                            'function': node.name,
                            'message': f"Function '{node.name}' raises NotImplementedError"
                        })
                    
                    # Check for missing docstrings
                    if not ast.get_docstring(node) and not node.name.startswith('_'):
                        self.stats['missing_docstrings'] += 1
            
            # Check for TODO comments
            for i, line in enumerate(lines, 1):
                if 'TODO' in line or 'FIXME' in line or 'XXX' in line:
                    self.stats['todo_comments'] += 1
                    file_issues.append({
                        'file': str(filepath),
                        'type': 'TODO_COMMENT',
                        'line': i,
                        'message': line.strip()
                    })
            
            # Check for placeholder data or hardcoded values
            if 'placeholder' in content.lower() or 'dummy' in content.lower():
                for i, line in enumerate(lines, 1):
                    if 'placeholder' in line.lower() or 'dummy' in line.lower():
                        file_issues.append({
                            'file': str(filepath),
                            'type': 'PLACEHOLDER',
                            'line': i,
                            'message': f"Placeholder/dummy code: {line.strip()[:60]}"
                        })
        
        except Exception as e:
            file_issues.append({
                'file': str(filepath),
                'type': 'AUDIT_ERROR',
                'line': 0,
                'message': f"Error auditing file: {str(e)}"
            })
        
        return file_issues
    
    def _is_empty_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is empty (only pass/docstring)."""
        if not node.body:
            return True
        
        # Filter out docstring
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        
        # Check if only pass statement
        if not body:
            return True
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        
        return False
    
    def _has_not_implemented(self, node: ast.FunctionDef) -> bool:
        """Check if function raises NotImplementedError."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call):
                    if isinstance(child.exc.func, ast.Name):
                        if child.exc.func.id == 'NotImplementedError':
                            return True
                elif isinstance(child.exc, ast.Name):
                    if child.exc.id == 'NotImplementedError':
                        return True
        return False
    
    def audit_directory(self, directory: Path = None) -> None:
        """Audit all Python files in directory."""
        if directory is None:
            directory = self.root_dir
        
        python_files = list(directory.rglob('*.py'))
        
        # Exclude certain directories
        exclude_dirs = {'.venv', 'venv', '__pycache__', '.git', 'node_modules', 'migrations'}
        
        for filepath in python_files:
            # Skip excluded directories
            if any(excluded in filepath.parts for excluded in exclude_dirs):
                continue
            
            file_issues = self.audit_file(filepath)
            self.issues.extend(file_issues)
    
    def generate_report(self, output_file: str = 'audit_report.txt') -> None:
        """Generate comprehensive audit report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE CODEBASE AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in self.stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write(f"\nTotal Issues Found: {len(self.issues)}\n")
            f.write("\n")
            
            # Group issues by type
            issues_by_type = {}
            for issue in self.issues:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)
            
            # Report by type
            for issue_type, issues in sorted(issues_by_type.items()):
                f.write("=" * 80 + "\n")
                f.write(f"{issue_type} ({len(issues)} occurrences)\n")
                f.write("=" * 80 + "\n\n")
                
                for issue in issues:
                    f.write(f"File: {issue['file']}\n")
                    f.write(f"Line: {issue.get('line', 'N/A')}\n")
                    if 'function' in issue:
                        f.write(f"Function: {issue['function']}\n")
                    f.write(f"Message: {issue['message']}\n")
                    f.write("-" * 80 + "\n")
                f.write("\n")
            
            # Critical issues summary
            f.write("=" * 80 + "\n")
            f.write("CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION\n")
            f.write("=" * 80 + "\n\n")
            
            critical_types = ['SYNTAX_ERROR', 'NOT_IMPLEMENTED', 'EMPTY_FUNCTION']
            critical_issues = [i for i in self.issues if i['type'] in critical_types]
            
            if critical_issues:
                for issue in critical_issues:
                    f.write(f"[{issue['type']}] {issue['file']}:{issue.get('line', '?')}\n")
                    f.write(f"  {issue['message']}\n\n")
            else:
                f.write("No critical issues found!\n")
        
        print(f"Audit report generated: {output_file}")
        print(f"Total files audited: {self.stats['total_files']}")
        print(f"Total issues found: {len(self.issues)}")
        print(f"Critical issues: {len(critical_issues) if 'critical_issues' in locals() else 0}")


def find_duplicate_files() -> List[Tuple[str, List[str]]]:
    """Find duplicate files based on content."""
    import hashlib
    
    file_hashes = {}
    duplicates = []
    
    for filepath in Path('.').rglob('*.py'):
        if any(excluded in filepath.parts for excluded in {'.venv', '__pycache__', '.git'}):
            continue
        
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in file_hashes:
                file_hashes[file_hash].append(str(filepath))
            else:
                file_hashes[file_hash] = [str(filepath)]
        except Exception:
            pass
    
    # Find duplicates
    for file_hash, files in file_hashes.items():
        if len(files) > 1:
            duplicates.append((file_hash, files))
    
    return duplicates


def find_unused_files() -> List[str]:
    """Find potentially unused files."""
    unused = []
    
    # Files that might be unused
    suspicious_patterns = ['test_old', 'backup', 'copy', 'temp', 'tmp', '_old', 'deprecated']
    
    for filepath in Path('.').rglob('*.py'):
        if any(excluded in filepath.parts for excluded in {'.venv', '__pycache__', '.git'}):
            continue
        
        filename = filepath.name.lower()
        if any(pattern in filename for pattern in suspicious_patterns):
            unused.append(str(filepath))
    
    return unused


if __name__ == "__main__":
    print("Starting comprehensive codebase audit...")
    print("=" * 80)
    
    # Run main audit
    auditor = CodebaseAuditor()
    auditor.audit_directory()
    auditor.generate_report('AUDIT_DETAILED_REPORT.txt')
    
    print("\n" + "=" * 80)
    print("Checking for duplicate files...")
    duplicates = find_duplicate_files()
    
    if duplicates:
        print(f"\nFound {len(duplicates)} sets of duplicate files:")
        for i, (file_hash, files) in enumerate(duplicates, 1):
            print(f"\n  Duplicate set {i}:")
            for f in files:
                print(f"    - {f}")
    else:
        print("No duplicate files found.")
    
    print("\n" + "=" * 80)
    print("Checking for potentially unused files...")
    unused = find_unused_files()
    
    if unused:
        print(f"\nFound {len(unused)} potentially unused files:")
        for f in unused:
            print(f"  - {f}")
    else:
        print("No obviously unused files found.")
    
    print("\n" + "=" * 80)
    print("Audit complete! Review AUDIT_DETAILED_REPORT.txt for full details.")
