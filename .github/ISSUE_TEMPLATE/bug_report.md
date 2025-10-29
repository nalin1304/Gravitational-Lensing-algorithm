---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior:
1. Go to '...'
2. Run command '....'
3. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened instead.

## Error Messages

```
Paste any error messages or stack traces here
```

## Environment

- OS: [e.g. Windows 11, Ubuntu 22.04, macOS 13.0]
- Python version: [e.g. 3.10.0]
- Package versions: (run `pip list | grep -E "torch|numpy|scipy"`)
- GPU: [Yes/No, if yes specify model]

## Minimal Reproducible Example

```python
# Please provide minimal code that reproduces the issue
import sys
sys.path.append('..')
from src.lens_models import NFWProfile

# Your code here...
```

## Additional Context

Add any other context about the problem here (screenshots, data files, etc.).

## Possible Solution

If you have suggestions on how to fix the bug, please describe them here.
