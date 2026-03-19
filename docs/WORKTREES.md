# Git Worktrees in Claude Code — A Complete Guide

## Table of Contents

1. [What Are Git Worktrees?](#what-are-git-worktrees)
2. [The 3 Worktrees in This Project](#the-3-worktrees-in-this-project)
3. [How Worktrees Work Under the Hood](#how-worktrees-work-under-the-hood)
4. [Where Are the Worktree Directories?](#where-are-the-worktree-directories)
5. [How to View and Test Each Worktree Before Merging](#how-to-view-and-test-each-worktree-before-merging)
6. [How the Merge Works](#how-the-merge-works)
7. [Are Worktrees Preserved?](#are-worktrees-preserved)
8. [Invoking Worktrees in Claude Code](#invoking-worktrees-in-claude-code)
9. [Pros and Cons of Worktrees in Claude Code](#pros-and-cons-of-worktrees-in-claude-code)
10. [Worktree Lifecycle Summary](#worktree-lifecycle-summary)

---

## What Are Git Worktrees?

A **git worktree** is a separate working directory that shares the same `.git` repository as your main checkout. Think of it as having multiple copies of your project checked out simultaneously, each on a different branch, but all sharing the same git history.

Without worktrees, you'd need to:
- Stash your changes, switch branches, make edits, commit, switch back, pop stash — tedious and error-prone
- Or clone the repo multiple times — wastes disk space and creates divergent histories

With worktrees, each directory is an independent workspace. You can edit, build, and test in each one without affecting the others. They all share the same `.git` database, so commits made in any worktree are immediately visible to all others.

**In Claude Code**, worktrees enable **parallel development**: Claude launches multiple agents, each working in its own worktree on a separate branch, editing files simultaneously without conflicts. This is how the 3 features below were developed in parallel in a single conversation.

---

## The 3 Worktrees in This Project

Three agents were launched in parallel, each in its own worktree, each tackling an independent task:

### Worktree 1: Frontend Redesign

| Property | Value |
|----------|-------|
| **Branch** | `worktree-agent-a600b596` |
| **Directory** | `.claude/worktrees/agent-a600b596/` |
| **Commit** | `9744d50` — "Redesign landing page with terminal-themed dashboard and sidebar branding" |
| **Files changed** | `app.py` (+525 lines) |

**What it does**: Transforms the plain landing page (the screen you see before clicking "Run Analysis") into a polished, Bloomberg/terminal-inspired dark-themed dashboard.

**Changes in detail**:
- **Global CSS block** (~200 lines): Custom styles injected via `st.markdown(unsafe_allow_html=True)`. Imports JetBrains Mono (code font) and DM Sans (body font). Defines CSS variables for a green/cyan-on-dark color palette. Styles cards with box-shadows, gradients, and hover lift effects.
- **Sidebar branding**: SVG logo (mini line chart icon), "HMM REGIME / Terminal" wordmark, version badge — all at the top of the sidebar before any settings.
- **Landing page** (replaces the single `st.info()` line in the `else` block):
  - Hero section with gradient-accented title and subtitle
  - 5-step pipeline visualization: Fetch Data → Engineer Features → Fit HMM → Detect Regimes → Generate Signals
  - Math concept cards in a responsive grid (HMM, BIC, Shannon Entropy, Kelly Criterion, Walk-Forward Validation) — each with formula snippets and color-coded top borders
  - Getting started guide directing users to the sidebar
  - Footer with key algorithm names

**What it does NOT touch**: All existing analysis functionality (the `if run_btn:` block) is completely untouched. Only the `else` block and CSS were modified.

---

### Worktree 2: Fundamental Analysis

| Property | Value |
|----------|-------|
| **Branch** | `worktree-agent-a8cdba18` |
| **Directory** | `.claude/worktrees/agent-a8cdba18/` |
| **Commit** | `2cde3b1` — "Add fundamental analysis module and Fundamentals tab" |
| **Files changed** | `fundamentals.py` (new, 376 lines), `app.py` (+314 lines) |

**What it does**: Adds a complete fundamental analysis module and a new "Fundamentals" tab (Tab 6) to the dashboard.

**New file — `fundamentals.py`**:
- `FundamentalAnalyzer` class using yfinance's `Ticker` API
- `get_company_overview(ticker)` — name, sector, industry, market cap, description, 52-week range
- `get_financial_ratios(ticker)` — P/E, P/B, P/S, PEG, EV/EBITDA, debt-to-equity, current ratio, ROE, ROA, margins, dividend info, beta
- `get_financial_statements(ticker)` — income statement, balance sheet, cash flow (DataFrames)
- `get_analyst_data(ticker)` — recommendations, price targets, earnings dates
- `format_large_number(n)` — human-readable formatting (e.g., "1.2B", "340M")
- `health_score(ratios)` — composite 0–100 score across profitability, valuation, liquidity, leverage, growth
- `is_crypto(ticker)` — detects crypto tickers (shows "not available" message instead of attempting fundamental analysis)

**Dashboard changes — Tab 6 "Fundamentals"**:
- Company overview card (name, sector, market cap, description)
- Financial health score gauge (Plotly indicator)
- Color-coded financial ratios grid (green/yellow/red)
- Income statement grouped bar chart (revenue + net income over time)
- Balance sheet composition chart
- Cash flow summary chart
- Analyst consensus section (buy/hold/sell counts, price target vs current price)
- Expandable sections for recent recommendations and earnings dates
- Crypto tickers show a graceful warning instead of errors

---

### Worktree 3: Comprehensive Documentation

| Property | Value |
|----------|-------|
| **Branch** | `worktree-agent-ac031831` |
| **Directory** | `.claude/worktrees/agent-ac031831/` |
| **Commit** | `dd50bdf` — "Rewrite and expand documentation with theory, implementation, and user guides" |
| **Files changed** | 5 files (+3,859 lines, -734 lines rewritten) |

**What it does**: Rewrites existing docs and creates 2 new comprehensive documents.

**Rewritten files**:
- `docs/ARCHITECTURE.md` (~800+ lines) — Full system diagram, module dependency graph, three data flow diagrams (full pipeline, walk-forward loop, signal state machine), configuration cascade, Streamlit state/caching analysis, threading/performance model, error handling strategy
- `docs/USER_GUIDE.md` (~700+ lines) — Platform-specific venv instructions (Git Bash, cmd, PowerShell, Linux, Mac), first-run walkthrough, annotated tab tour, parameter tuning recipes, 5 workflows, troubleshooting, FAQ, 30+ term glossary
- `README.md` — Added Documentation section with links to all docs

**New files**:
- `docs/THEORY.md` (~600 lines) — Formal HMM definition, Baum-Welch EM derivation (alpha/beta/xi/gamma), Viterbi with worked example, BIC from Bayesian model comparison, Shannon entropy, expected regime duration, stationary distribution, Kelly criterion derivation, CVaR, bootstrap CIs, walk-forward validation, feature engineering rationale. Uses LaTeX `$...$` notation.
- `docs/IMPLEMENTATION.md` (~500 lines) — Module-by-module code walkthrough, hmmlearn API details, random restart strategy, signal generation edge cases, trade simulation mechanics, bootstrap implementation, performance profiling, known limitations, testing strategy

---

## How Worktrees Work Under the Hood

### The Git Mechanics

When Claude Code creates a worktree, it runs:

```bash
git worktree add .claude/worktrees/agent-XXXX -b worktree-agent-XXXX
```

This does 3 things:
1. **Creates a new directory** at `.claude/worktrees/agent-XXXX/` with a full checkout of the repo
2. **Creates a new branch** called `worktree-agent-XXXX` starting from the current HEAD of `main`
3. **Links the worktree** to the main repo's `.git` directory (via a `.git` file in the worktree that points back)

The key insight: **all worktrees share the same `.git` database**. Commits, branches, tags, and objects are shared. Only the working directory and index (staging area) are separate.

### What the Agent Does Inside

Each agent:
1. Receives its task prompt
2. Reads files from its worktree directory (which starts as a copy of `main`)
3. Edits files using the same tools (Read, Edit, Write) but scoped to the worktree path
4. Commits changes to its own branch
5. Returns a summary to the main conversation

The agents run **concurrently** — all 3 were working at the same time. This is only possible because each has its own working directory. Without worktrees, they'd be fighting over the same files.

### The Shared State

```
.git/                          ← shared git database (one copy)
├── objects/                   ← all commits, trees, blobs
├── refs/heads/
│   ├── main                   ← d9e0959
│   ├── worktree-agent-a600b596 ← 9744d50
│   ├── worktree-agent-a8cdba18 ← 2cde3b1
│   └── worktree-agent-ac031831 ← dd50bdf
└── worktrees/
    ├── agent-a600b596/        ← worktree metadata
    ├── agent-a8cdba18/
    └── agent-ac031831/

Main checkout:     C:/Users/simon/Downloads/projects/HMM/          → main
Worktree 1:        .claude/worktrees/agent-a600b596/                → worktree-agent-a600b596
Worktree 2:        .claude/worktrees/agent-a8cdba18/                → worktree-agent-a8cdba18
Worktree 3:        .claude/worktrees/agent-ac031831/                → worktree-agent-ac031831
```

---

## Where Are the Worktree Directories?

All worktrees live under `.claude/worktrees/` inside your project:

```
C:\Users\simon\Downloads\projects\HMM\
├── .claude\
│   └── worktrees\
│       ├── agent-a600b596\      ← Worktree 1 (frontend redesign)
│       │   ├── app.py           ← modified
│       │   ├── config.yaml      ← unchanged (from main)
│       │   ├── data_loader.py   ← unchanged
│       │   └── ...
│       ├── agent-a8cdba18\      ← Worktree 2 (fundamentals)
│       │   ├── app.py           ← modified
│       │   ├── fundamentals.py  ← NEW file
│       │   └── ...
│       └── agent-ac031831\      ← Worktree 3 (documentation)
│           ├── README.md        ← modified
│           ├── docs\
│           │   ├── ARCHITECTURE.md  ← rewritten
│           │   ├── THEORY.md        ← NEW
│           │   ├── IMPLEMENTATION.md ← NEW
│           │   └── USER_GUIDE.md    ← rewritten
│           └── ...
├── app.py                       ← main branch (unchanged)
├── config.yaml
└── ...
```

Each worktree directory is a **complete checkout** — it has every file from `main`, plus its modifications. You can `cd` into any worktree and it works exactly like a normal git repo.

To list all worktrees:

```bash
git worktree list
```

Output:
```
C:/Users/simon/Downloads/projects/HMM                                  d9e0959 [main]
C:/Users/simon/Downloads/projects/HMM/.claude/worktrees/agent-a600b596 9744d50 [worktree-agent-a600b596]
C:/Users/simon/Downloads/projects/HMM/.claude/worktrees/agent-a8cdba18 2cde3b1 [worktree-agent-a8cdba18]
C:/Users/simon/Downloads/projects/HMM/.claude/worktrees/agent-ac031831 dd50bdf [worktree-agent-ac031831]
```

---

## How to View and Test Each Worktree Before Merging

This is the biggest advantage of worktrees — you can **test each change in isolation** before deciding to merge.

### Option A: Run the App from a Worktree Directory

Each worktree is a fully functional checkout. You can run the app directly from any of them:

```bash
# Test the frontend redesign
cd .claude/worktrees/agent-a600b596
python -m streamlit run app.py
# → See the new landing page at http://localhost:8501

# Test fundamental analysis (use a different port to run simultaneously)
cd .claude/worktrees/agent-a8cdba18
python -m streamlit run app.py --server.port 8502
# → See the new Fundamentals tab at http://localhost:8502

# Test documentation (just read the files)
cd .claude/worktrees/agent-ac031831
# Browse docs/THEORY.md, docs/IMPLEMENTATION.md, etc.
```

Note: You'll need to install dependencies in each worktree (or use the same venv):

```bash
cd .claude/worktrees/agent-a600b596
source ../../.venv/Scripts/activate    # reuse the main venv
python -m streamlit run app.py
```

### Option B: Review the Code Diff

See exactly what changed in each worktree without running anything:

```bash
# What did the frontend redesign change?
cd .claude/worktrees/agent-a600b596
git diff main

# What did the fundamentals feature add?
cd .claude/worktrees/agent-a8cdba18
git diff main

# What did the docs rewrite change?
cd .claude/worktrees/agent-ac031831
git diff main

# Or from the main directory, compare branches:
git diff main..worktree-agent-a600b596 -- app.py
git diff main..worktree-agent-a8cdba18 --stat
git diff main..worktree-agent-ac031831 --stat
```

### Option C: Cherry-Pick What You Like

If you only want some changes, cherry-pick individual commits:

```bash
# Only take the frontend redesign
git cherry-pick 9744d50

# Only take the fundamentals module
git cherry-pick 2cde3b1
```

### Option D: Create Pull Requests

Push each worktree branch to GitHub and review via pull requests:

```bash
git push origin worktree-agent-a600b596
git push origin worktree-agent-a8cdba18
git push origin worktree-agent-ac031831
# Then create PRs on GitHub for code review
```

---

## How the Merge Works

### Conflict Analysis

Before merging, it's important to understand which files each worktree touched:

| File | Worktree 1 (Frontend) | Worktree 2 (Fundamentals) | Worktree 3 (Docs) |
|------|-----------------------|---------------------------|---------------------|
| `app.py` | MODIFIED (CSS + landing) | MODIFIED (Tab 6) | not touched |
| `fundamentals.py` | — | NEW | — |
| `README.md` | — | — | MODIFIED |
| `docs/ARCHITECTURE.md` | — | — | REWRITTEN |
| `docs/USER_GUIDE.md` | — | — | REWRITTEN |
| `docs/THEORY.md` | — | — | NEW |
| `docs/IMPLEMENTATION.md` | — | — | NEW |

**Conflict risk**: Worktrees 1 and 2 both modify `app.py`, but in different sections:
- Worktree 1 modifies the CSS and the `else` block (landing page)
- Worktree 2 adds a new tab inside the `if run_btn:` block and modifies the tab list

Git may or may not auto-resolve this depending on how close the edits are. If there's a conflict, it will be in `app.py` only. Worktree 3 doesn't touch any code files, so it merges cleanly.

### Merge Strategy

**Recommended order** (least conflict risk first):

```bash
# Step 1: Merge documentation (no code conflicts possible)
git merge worktree-agent-ac031831 -m "Merge comprehensive documentation rewrite"

# Step 2: Merge fundamentals (adds new file + tab)
git merge worktree-agent-a8cdba18 -m "Merge fundamental analysis module"

# Step 3: Merge frontend (may conflict with step 2 on app.py)
git merge worktree-agent-a600b596 -m "Merge frontend redesign"
# If conflict: resolve in app.py, then git add app.py && git commit
```

### If Conflicts Occur

When merging Worktrees 1 and 2, git will mark conflicts in `app.py` like this:

```python
<<<<<<< HEAD
# ... code from the branch you already merged
=======
# ... code from the branch you're merging now
>>>>>>> worktree-agent-a600b596
```

Open `app.py`, find the conflict markers, keep both sets of changes (they modify different parts), remove the markers, then:

```bash
git add app.py
git commit -m "Resolve merge conflict: combine frontend redesign with fundamentals tab"
```

---

## Are Worktrees Preserved?

### Default Behavior

Claude Code **preserves worktrees that have changes**. If an agent makes no changes (nothing to commit), the worktree is automatically cleaned up. Since all 3 agents committed changes, all 3 worktrees still exist.

### After Merging

After you merge the branches into `main`, the worktree directories **still exist** on disk until you explicitly remove them:

```bash
# Remove a specific worktree (after merging its branch)
git worktree remove .claude/worktrees/agent-a600b596

# Or remove all three
git worktree remove .claude/worktrees/agent-a600b596
git worktree remove .claude/worktrees/agent-a8cdba18
git worktree remove .claude/worktrees/agent-ac031831

# Optionally delete the merged branches too
git branch -d worktree-agent-a600b596
git branch -d worktree-agent-a8cdba18
git branch -d worktree-agent-ac031831
```

### If You Don't Clean Up

No harm done. The worktree directories are inside `.claude/worktrees/` which is typically in `.gitignore`, so they won't be committed. They just take up disk space (a full checkout each, minus the shared `.git` objects).

---

## Invoking Worktrees in Claude Code

You don't need to know any git commands. Just describe what you want in natural language, and mention that you want parallel/isolated work.

### Natural Language Patterns That Trigger Worktrees

```
"Start 3 worktrees: one for X, one for Y, one for Z"

"Use worktrees to work on these in parallel"

"In separate worktrees, do A and B"

"Create a worktree to experiment with refactoring the backend"

"Launch agents in worktrees for each of these tasks"
```

### Examples from This Project

What was actually said to create these 3 worktrees:

> "start 3 git worktrees: 1. first worktree use frontend-design skill to improve the landing page. 2. second worktree adds fundamental analysis with data also from yahoo finance, 3. 3rd worktree writes extensive documentations"

Claude Code then:
1. Launched 3 Agent tool calls in the same message, each with `isolation: "worktree"`
2. Each agent received a detailed prompt describing its task
3. All 3 ran concurrently in their own directories
4. Results were reported back when all 3 completed

### Single Worktree for Experimental Changes

You can also use a single worktree for risky changes you might want to discard:

```
"Try refactoring the HMM engine in a worktree — I want to review before merging"

"Experiment with switching from hmmlearn to pomegranate in a worktree"
```

This keeps your `main` branch clean while Claude experiments.

---

## Pros and Cons of Worktrees in Claude Code

### Pros

| Advantage | Explanation |
|-----------|-------------|
| **True parallelism** | Multiple agents edit different files simultaneously. A 3-task job that takes 15 minutes sequentially completes in 5 minutes (wall clock). In this project, all 3 agents ran concurrently. |
| **Isolation** | Each agent works in its own directory. No risk of Agent A's half-finished edits breaking Agent B's work. If one agent fails, the others are unaffected. |
| **Review before merge** | You can inspect, test, and run each worktree's changes independently before deciding to merge. You can accept some and reject others. |
| **Clean git history** | Each feature gets its own branch and commit. You can merge with `--squash` for a single commit, or keep the full history. |
| **No stashing needed** | Your main working directory stays exactly as it was. No stash/pop juggling. |
| **Rollback is trivial** | Don't like a worktree's changes? Just `git worktree remove` it and delete the branch. Zero impact on main. |
| **Disk-efficient** | Worktrees share the `.git` object store. Each worktree only costs the size of the working directory files, not a full clone. |

### Cons

| Disadvantage | Explanation |
|--------------|-------------|
| **Merge conflicts** | If multiple worktrees edit the same file (like `app.py` in Worktrees 1 and 2), you'll need to resolve conflicts manually when merging. |
| **Context duplication** | Each agent starts fresh — it doesn't know what the other agents are doing. If tasks are interdependent (Agent B needs Agent A's output), worktrees aren't the right tool. |
| **Disk space** | Each worktree is a full checkout. For large repos with big binary files, this adds up. For this project (~10MB), it's negligible. |
| **Dependency management** | Each worktree may need its own venv, or you need to carefully share one. New dependencies added in one worktree won't be available in others. |
| **Cleanup overhead** | After merging, you need to manually remove worktrees and branches. Not automatic. |
| **Harder to coordinate** | If you realize mid-way that Agent A should import a module Agent B is creating, there's no way to communicate between running agents. Plan the division of labor carefully upfront. |
| **Cross-feature blindness** | Each agent has zero knowledge of what others are building. If Agent A creates a UI that should reference Agent B's new feature, it won't — because it doesn't know the feature exists. Requires a post-merge integration pass. See [The Cross-Feature Awareness Problem](#the-cross-feature-awareness-problem). |
| **Merge order matters** | With 3 worktrees touching overlapping files, the order you merge affects which conflicts arise. Plan the merge sequence. |

### The Cross-Feature Awareness Problem

This is the most important limitation to understand. It was discovered during this project's development and is documented here as a lesson learned.

**What happened:** Three worktrees were launched in parallel:
- Worktree 1: Redesign the landing page (pipeline visualization, concept cards)
- Worktree 2: Add a Fundamentals tab with financial analysis
- Worktree 3: Rewrite all documentation

After merging all three, the landing page (from Worktree 1) had **no mention of the Fundamentals feature** (from Worktree 2). The pipeline showed only 5 steps instead of 6. The concept cards had 5 cards instead of 6. The hero subtitle didn't mention fundamental analysis.

**Why it happened:** Each agent starts from the same `main` HEAD and has zero knowledge of what the other agents are building. Worktree 1 designed a landing page for a 5-tab app because that's what existed when it started. Worktree 2 added Tab 6, but Worktree 1 never knew about it.

**The fix required a post-merge integration commit** to add the missing pipeline step, concept card, and subtitle text.

#### How to Prevent This

**Strategy 1: Shared spec in the prompt (recommended)**

When launching worktrees, tell each agent about the others' work:

```
"Start 3 worktrees:
1. Redesign the landing page. NOTE: the app will have 6 tabs
   including a new Fundamentals tab (financial ratios, health score,
   analyst data) being built by another agent. Include it in the
   pipeline and concept cards.
2. Add fundamental analysis module and Fundamentals tab.
3. Rewrite documentation. NOTE: include the new Fundamentals tab
   in all relevant sections."
```

The agents still can't see each other's code, but they can design for features they know are coming. This prevents most gaps at zero cost.

**Strategy 2: Post-merge integration pass**

Accept that gaps will happen and plan for a review after merging:

```
"Review app.py for consistency — make sure the landing page,
tabs, and docs all reflect every feature that was merged."
```

**Strategy 3: Split by layer, not feature**

Instead of one agent per feature, split by concern:
- Agent 1: All backend modules (new .py files, data logic)
- Agent 2: All frontend/UI (app.py — all tabs, landing page, CSS)
- Agent 3: All documentation

This way the UI agent owns the full interface and won't miss a tab. The tradeoff is that the UI agent needs to wait for the backend agent's API to be defined (or receive a spec).

**Strategy 4: Sequential for coupled features**

If feature B should be reflected in feature A's UI, don't parallelize them. Run B first, merge it, then run A so it sees B's work. Reserve parallelism for truly independent work.

**Best practice: Strategy 1 + Strategy 2.** Brief each agent on what the others are building (prevents most gaps), then do a quick integration check after merge (catches the rest).

### When to Use Worktrees

**Good fit**:
- Independent features (different files or different sections of the same file)
- Experimental changes you want to review before committing
- Documentation + code work in parallel (docs rarely conflict with code)
- Multiple bug fixes in unrelated parts of the codebase

**Bad fit**:
- Sequential tasks (B depends on A's output)
- Tightly coupled changes to the same file/function
- Quick single-file edits (overhead not worth it)
- Tasks that need to share runtime state or test results

---

## Worktree Lifecycle Summary

```
 User prompt
    │
    ▼
 Claude Code creates N agents with isolation: "worktree"
    │
    ├── git worktree add .claude/worktrees/agent-XXXX -b worktree-agent-XXXX
    ├── git worktree add .claude/worktrees/agent-YYYY -b worktree-agent-YYYY
    └── git worktree add .claude/worktrees/agent-ZZZZ -b worktree-agent-ZZZZ
    │
    ▼
 Agents run concurrently (read, edit, write, commit in their own directories)
    │
    ▼
 Agents return results to main conversation
    │
    ▼
 User reviews each worktree:
    ├── cd .claude/worktrees/agent-XXXX && git diff main    (review code)
    ├── python -m streamlit run app.py                      (test app)
    └── git diff main..worktree-agent-XXXX --stat           (see scope)
    │
    ▼
 User merges (or discards):
    ├── git merge worktree-agent-XXXX     (accept)
    ├── git cherry-pick COMMIT_HASH       (accept partially)
    └── git branch -D worktree-agent-YYYY (reject)
    │
    ▼
 Cleanup:
    ├── git worktree remove .claude/worktrees/agent-XXXX
    ├── git worktree remove .claude/worktrees/agent-YYYY
    ├── git worktree remove .claude/worktrees/agent-ZZZZ
    └── git branch -d worktree-agent-XXXX worktree-agent-YYYY worktree-agent-ZZZZ
```
