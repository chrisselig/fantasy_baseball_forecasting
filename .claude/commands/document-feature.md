You are documenting a feature called: $ARGUMENTS

## Step 1 — Analyze the codebase

Search the codebase for all files related to "$ARGUMENTS". Look for:
- Components, modules, or files whose names match the feature
- Functions, classes, or exports that implement the feature
- Types and interfaces introduced for it
- Any existing docs in `docs/` that relate to it
- Test files covering it

From your analysis, determine the feature type:
- **Frontend** — only UI components, no backend API calls
- **Backend** — only server-side logic, data models, or API routes
- **Full-stack** — both UI and server-side code involved

Note the feature type — it will shape the documentation sections below.

---

## Step 2 — Create `docs/dev/$ARGUMENTS.md`

Generate a technical reference for developers. Adjust sections based on feature type.

```markdown
# [Feature Name] — Developer Documentation

> **Related user guide:** [How to use [Feature Name]](../user/$ARGUMENTS.md)

## Overview
[1–2 sentences: what the feature does and why it was built]

## Feature Type
[Frontend / Backend / Full-stack] — [one sentence explaining scope]

## Files Added / Modified
| File | Change | Purpose |
|------|--------|---------|
| ... | Added / Modified | ... |

## Architecture
[Describe the structure in plain text or ASCII diagram.
For full-stack features, show the request/response flow.
For frontend, show the component tree and data flow.
For backend, show the service/data layer breakdown.]

## API / Interface Reference

### Components (frontend/full-stack only)
For each component:
**`<ComponentName>`**
| Prop | Type | Required | Description |
|------|------|----------|-------------|
| ... | ... | ... | ... |

### Functions & Exports
For each exported function:
**`functionName(param: Type): ReturnType`**
[Description, parameter details, return value, side effects]

### TypeScript Types
[List and explain all new types and interfaces]

### API Endpoints (backend/full-stack only)
For each endpoint:
**`METHOD /path`**
- Auth required: yes/no
- Request body: [shape]
- Response: [shape]
- Error codes: [list]

## State Management
[Where state lives, how it flows, what persists (localStorage, DB, etc.)]

## Error Handling
[What is handled, how, and where there are known gaps]

## Security Considerations
[Input validation, XSS risks, auth checks, data exposure]

## Extending This Feature
[How to add new options, formats, integrations — and what to watch out for]

## Known Limitations
[Edge cases not handled, browser/platform constraints, performance notes]
```

---

## Step 3 — Create `docs/user/$ARGUMENTS.md`

Generate a plain-language guide for end users. Write simply and encouragingly.

```markdown
# How to Use [Feature Name]

> **Technical details for developers:** [$ARGUMENTS Implementation](../dev/$ARGUMENTS.md)

## What is [Feature Name]?
[1–2 sentences explaining what this feature does and why it helps the user. No jargon.]

## Before You Start
[Any prerequisites — being logged in, having data set up, permissions needed, etc. Skip if none.]

## How to Use It

### Step 1 — [Action title]
[Plain-language instruction]

![Step 1 — [description of what the screenshot shows]](screenshots/$ARGUMENTS-step-1.png)

### Step 2 — [Action title]
[Plain-language instruction]

![Step 2 — [description of what the screenshot shows]](screenshots/$ARGUMENTS-step-2.png)

[Continue for all steps...]

## Your Options Explained
| Option | What it does | When to use it |
|--------|-------------|----------------|
| ... | ... | ... |

## Tips & Tricks
- [Practical tip 1]
- [Practical tip 2]
- [Practical tip 3]

## Related Features
[Link to any related user guides that already exist in docs/user/]

## Troubleshooting

**[Most likely problem]**
[Solution in plain language]

**[Second most likely problem]**
[Solution in plain language]

**[Third most likely problem]**
[Solution in plain language]
```

---

## Step 4 — Check for existing docs to cross-reference

Scan `docs/` for any existing documentation files. If any relate to "$ARGUMENTS", add links to them in both generated files under a "Related Documentation" section at the bottom.

---

## Step 5 — Print a summary

After creating both files, output:

```
Documentation generated for: $ARGUMENTS
Feature type detected: [Frontend / Backend / Full-stack]

Files created:
  docs/dev/$ARGUMENTS.md
  docs/user/$ARGUMENTS.md

Screenshot placeholders added: [N] — create these at docs/user/screenshots/
Cross-references added: [list any existing docs linked]

Implementation notes:
[Flag any gaps, missing error handling, security concerns, or incomplete features found during analysis]
```
