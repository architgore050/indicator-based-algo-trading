import re
import sys

with open(r'C:\Users\gorea\Desktop\indicator based algo trading\indicator based algo trading\docs\test.txt', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

lines = content.split('\n')
print(f"Total lines: {len(lines)}")
print(f"Total chars: {len(content)}")

# Search for constraint-related messages
constraint_patterns = ['constraint', 'min_trades', 'max_trades', 'penalty_weight', 'max_max_drawdown', 'trades_per_day']
constraint_matches = []
for i, line in enumerate(lines):
    for pat in constraint_patterns:
        if pat.lower() in line.lower():
            constraint_matches.append((i+1, line.strip()))
            break

print(f"\nConstraint-related lines: {len(constraint_matches)}")
for line_no, text in constraint_matches[:20]:
    print(f"  Line {line_no}: {text[:200]}")

# Search for error patterns
error_patterns = ['ERROR', 'Traceback', 'Exception', 'Failed', 'exit status', 'ValueError', 'TypeError', 'KeyError']
error_matches = []
for i, line in enumerate(lines):
    for pat in error_patterns:
        if pat in line:
            error_matches.append((i+1, line.strip()))
            break

print(f"\nError-related lines: {len(error_matches)}")
for line_no, text in error_matches[:30]:
    print(f"  Line {line_no}: {text[:200]}")

# Check for completion message
print(f"\nContains 'WFV Pipeline Complete': {'WFV Pipeline Complete' in content}")
print(f"Contains 'Constraint violation': {'Constraint violation' in content}")

# Check head
print(f"\n--- FIRST 5 LINES ---")
for line in lines[:5]:
    print(f"  {line[:200]}")

# Check tail
print(f"\n--- LAST 10 LINES ---")
for line in lines[-10:]:
    print(f"  {line[:200]}")
