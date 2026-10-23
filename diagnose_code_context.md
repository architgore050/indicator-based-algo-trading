# diagnose_code_context.md

## When to Use

Before implementing any code change, bug fix, or refactoring that touches more than a single line. This is the "diagnose before you prescribe" phase — equivalent to a doctor running tests before prescribing medication.

**Trigger conditions:**
- The user says "diagnose this" or "understand the context first"
- The task involves changing config values, function signatures, or cross-file dependencies
- The user explicitly asks you to read and understand before acting
- The change could affect multiple files or have unexpected side effects

**Do NOT use for:** trivial fixes (typos, single-line corrections), adding new independent features, documentation updates, or when the user explicitly says "just do it."

---

## The Method

When asked to diagnose a code area, follow this sequence:

### 1. Read the Target Code

Read the specific file(s) and line range the user points to. Understand what the code does, not just what it looks like.

### 2. Trace All References

Search the entire codebase for every reference to the keys, variables, function names, and config values involved. Use `grep` to find:
- Where config keys are read
- Where function parameters are passed
- Where values are compared or used as defaults
- Where hardcoded values exist that should be config-driven

### 3. Map Dependencies

Build a mental model of:
- **What depends on this code?** (consumers, callers, scripts that read its output)
- **What does this code depend on?** (inputs, config, other functions it calls)
- **What behavior does this produce?** (side effects, files written, API responses, data transformations)
- **What would break if this changes?** (downstream scripts, parity tests, existing results)

### 4. Identify Hidden Patterns

Look for:
- Hardcoded values that should be config-driven (magic numbers, string literals, row counts)
- Stale references (config keys that exist in one file but not another, or vice versa)
- Naming mismatches (a key called `calibration_rows` that actually means "WFV start offset")
- Dead code (parameters sampled by Optuna but never consumed by signal generation)
- Duplicate logic (same pattern repeated across files that would benefit from centralization)
- Boundary conditions (what happens at row 0, at end of data, in smoke mode vs full mode)

### 5. Assess Impact

For each change needed, determine:
- **Scope:** Which files must change? Which files should NOT change (backups, parity tests, docs)?
- **Risk:** What could go wrong? (negative row indices, OOM, silent data loss, parity test failures)
- **Dependencies:** What other systems depend on the current behavior?
- **Verification:** How will we know the fix is correct? (smoke test, parity test, manual inspection)

### 6. Report Back

Present your findings to the user BEFORE implementing anything. Structure the report as:

```
## Diagnosis: [Brief Title]

### Current State
[What the code does now, with specific line references]

### Files That Reference [Key/Variable]
| File | Line | Usage | Impact if Changed |
|---|---|---|---|

### Hidden Patterns Found
- [Pattern 1]: [Description]
- [Pattern 2]: [Description]

### Proposed Changes
| File | Change | Why |
|---|---|---|

### Risks & Edge Cases
- [Risk 1]: [Mitigation]
- [Risk 2]: [Mitigation]

### Verification Plan
[How we'll test the fix]
```

### 7. Wait for Approval

Do NOT implement until the user reviews and approves the diagnosis. The user may:
- Approve the plan as-is
- Suggest modifications
- Reject certain changes
- Ask for deeper investigation into a specific area

---

## What "Thorough" Looks Like

A thorough diagnosis of a config change should answer:
1. Every file that reads this config key
2. Every file that writes/produces this config key
3. Every hardcoded value that mirrors this config key (and whether they match)
4. The semantic meaning of the value (what does `3000000` actually represent?)
5. What happens in smoke mode vs test mode vs full mode
6. What happens at boundary conditions (first window, last window, empty data)
7. Whether the naming matches the actual usage (or if there's a semantic mismatch)
8. What the downstream impact would be if the value changes

A thorough diagnosis of a function change should answer:
1. Every caller of this function
2. Every parameter and its current default
3. What the function returns and who consumes it
4. What side effects it produces
5. What assumptions it makes about its inputs
6. What would break if a parameter default changes
7. Whether the function has multiple code paths (smoke vs full, GPU vs CPU) and how they differ

---

## Anti-Patterns to Avoid

- **Don't skip reading.** Don't diagnose based on grep output alone. Read the actual code.
- **Don't over-engineer.** A thorough diagnosis is about understanding, not redesigning. Report what you find; don't propose architectural overhauls unless the user asks.
- **Don't implement without approval.** The diagnosis phase is separate from the implementation phase. Resist the urge to "just fix it."
- **Don't ignore parity tests.** If `.bak` files or parity test scripts exist, note them. They are the safety net for changes.
- **Don't guess.** If you're unsure about a value's meaning, say so. It's better to admit uncertainty than to make a wrong assumption.

---

## Example

**User:** "Diagnose the `calibration_rows` config key. I want to understand what it does and what would break if I change it."

**Agent should:**
1. Read `config.json` to see the key and its value
2. `grep` for all references to `calibration_rows` across the codebase
3. Read each referencing file to understand the usage context
4. Check for hardcoded values that match or contradict the config
5. Map out the data flow: who writes it, who reads it, how it's used
6. Report findings in the structured format above
7. Wait for user approval before making any changes

The user's prompt for Issue 8 was:
> "read the relevant code, diagnose the problem yourself, going beyond just what the action_items summarizes the problem as for you. read all relevant code in the full codebase (the part that we are to edit, how is this being used, what for, what will it affect, what depends on it showing certain behaviours, what other parts of the codebase does this part depend on, etc). then get back to me and I will tell you what to do next after you have fully understood the problem and its context / dependencies and constraints."

This skill captures that approach in a reusable, generalizable form.
