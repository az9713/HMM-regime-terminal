# Development Process — Parallel Worktree Development

This document describes how three major features were developed simultaneously
using git worktrees in Claude Code, and how the Chrome MCP extension was used
for visual UI testing.

## Overview

Three features were developed in parallel using git worktrees, each running
as an independent Claude Code agent with its own isolated copy of the repository:

| Worktree | Branch | Feature | Key Files |
|----------|--------|---------|-----------|
| 1 | `worktree-agent-a600b596` | Frontend redesign | `app.py` (+525 lines CSS/landing) |
| 2 | `worktree-agent-a8cdba18` | Fundamental analysis | `fundamentals.py` (new), `app.py` (Tab 6) |
| 3 | `worktree-agent-ac031831` | Documentation rewrite | 5 docs files (+3,859 lines) |

All three agents ran concurrently, completed independently, and were merged
into main in sequence.

---

## Worktree 1: Frontend Redesign

**Goal:** Transform the plain Streamlit landing page into a polished,
terminal-themed dashboard.

**What was built:**
- ~200 lines of custom CSS with a Bloomberg/terminal dark theme (JetBrains Mono
  font, green/cyan accent palette on dark surfaces)
- SVG logo and "HMM REGIME / Terminal" branded sidebar header with version badge
- Landing page hero section with gradient-accented title
- 5-step analysis pipeline visualization (Fetch Data -> Engineer Features ->
  Fit HMM -> Detect Regimes -> Generate Signals)
- Math concept cards grid (HMM, BIC, Shannon Entropy, Kelly Criterion,
  Walk-Forward Validation) with hover effects
- Getting started guide and footer

**Design review with Claude in Chrome:** After launching the worktree app on
a separate port (`localhost:8502`), the Claude in Chrome MCP extension was used
to take screenshots and inspect the rendered UI. This revealed contrast issues
with the signal display — the "SHORT" signal was hard to read on the dark
background. The fix: replaced `st.metric` with a styled HTML badge using
explicit `color:white` on a colored background, ensuring readability on both
light and dark themes.

For details on how the Chrome extension was used and an interesting routing
issue that was discovered, see:
- [MCP Chrome Explained](MCP_CHROME_EXPLAINED.md) — why Chrome MCP commands
  routed to the wrong machine
- [Browser Launch Bug](BROWSER_LAUNCH_BUG.md) — why `cmd.exe /c start` failed
  silently but direct `chrome.exe` invocation worked

---

## Worktree 2: Fundamental Analysis

**Goal:** Add fundamental financial analysis using Yahoo Finance data, with a
new dashboard tab.

**What was built:**

**`fundamentals.py`** — New module (376 lines) with `FundamentalAnalyzer` class:
- `get_company_overview()` — name, sector, industry, market cap, 52-week range
- `get_financial_ratios()` — P/E, P/B, P/S, PEG, EV/EBITDA, D/E, current ratio,
  ROE, ROA, margins, dividend, beta (18 ratios total)
- `get_financial_statements()` — income, balance sheet, cash flow summaries
- `get_analyst_data()` — recommendations, price targets, earnings dates
- `health_score()` — composite 0-100 score across profitability, valuation,
  liquidity, leverage, and growth
- `is_crypto()` — gracefully skips fundamental analysis for crypto tickers

**Tab 6 "Fundamentals"** in `app.py` — 314 new lines:
- Company overview card with sector, market cap, 52-week range
- Financial health score gauge (Plotly indicator, 0-100)
- Color-coded financial ratios grid (green/yellow/red based on value)
- Income statement trend chart (revenue + net income, grouped bars)
- Balance sheet composition chart (assets, liabilities, equity)
- Cash flow summary chart (operating, investing, financing, free)
- Analyst consensus section (recommendation badge, price target visualization)
- Expandable sections for detailed recommendations and earnings dates

---

## Worktree 3: Comprehensive Documentation

**Goal:** Rewrite existing docs and create new mathematical/implementation
references for complete project documentation.

**What was built:**

| Document | Lines | Description |
|----------|-------|-------------|
| [THEORY.md](THEORY.md) | 808 | Full mathematical reference: HMM definition, Baum-Welch derivation, Viterbi with worked example, BIC from Laplace approximation, Shannon entropy, Kelly criterion, CVaR, bootstrap CIs, walk-forward validation |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | 873 | Developer guide: module walkthroughs, hmmlearn API internals, random restart strategy, signal generation state machine, trade simulation, performance profiling, known limitations |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 1,481 | System architecture: data flow diagrams, module dependencies, configuration cascade, Streamlit state management, threading model, error handling |
| [USER_GUIDE.md](USER_GUIDE.md) | 1,377 | Complete onboarding: platform-specific install, first-run walkthrough, all tabs annotated, parameter tuning recipes, 5 workflows, troubleshooting, FAQ, 30+ term glossary |
| README.md | +54 | Added Documentation section with links to all docs |

---

## How Worktrees Were Used

### Invocation

All three worktrees were launched with a single natural language command:

> "start 3 git worktrees: 1. first worktree use frontend-design skill to improve
> the landing page. 2. second worktree adds fundamental analysis with data also
> from yahoo finance. 3. 3rd worktree writes extensive documentations"

Claude Code launched 3 agents simultaneously, each with `isolation: "worktree"`,
which created:
- 3 separate branches from the current `main` HEAD
- 3 separate working directories under `.claude/worktrees/`
- 3 independent agents, each reading/editing/committing in its own directory

### Testing Before Merge

Each worktree was tested independently before merging:

1. **Frontend (Worktree 1):** Launched on `localhost:8502`, inspected via Chrome
   MCP extension screenshots, contrast issues identified and fixed
2. **Fundamentals (Worktree 2):** Launched on `localhost:8502` (after stopping
   worktree 1), tested with AAPL ticker, verified Tab 6 renders correctly
3. **Documentation (Worktree 3):** Reviewed via `git diff` and file reads

### Merge Sequence

Merged in order of least conflict risk:
1. Documentation (worktree 3) — touched only docs, minor conflict in README.md
   and USER_GUIDE.md (resolved by taking worktree version)
2. Fundamentals (worktree 2) — added new file + new tab, merged cleanly
3. Frontend (worktree 1) — modified CSS and landing page, merged cleanly

For a complete guide to git worktrees in Claude Code, see
[WORKTREES.md](WORKTREES.md).

---

## Cross-Feature Awareness Gap (Lesson Learned)

### The Problem

After merging all three worktrees, a gap was discovered: the landing page
(from Worktree 1) had **no mention of the Fundamentals feature** (from
Worktree 2). Specifically:

- The pipeline visualization showed 5 steps instead of 6
- The concept cards section had 5 cards instead of 6
- The hero subtitle didn't mention fundamental analysis

This happened because each worktree agent started from the same `main` HEAD
and had zero knowledge of what the other agents were building. Worktree 1
designed a landing page for a 5-tab app — because that's all that existed
when it started. Worktree 2 added Tab 6, but Worktree 1 never knew.

### The Fix

A post-merge integration commit added:
- Step 06 "Fundamentals — Ratios, health score" to the pipeline
- A "Fundamental Analysis" concept card with P/E, ROE, D/E, Health Score
- "and fundamental financial analysis" to the hero subtitle

See commit `8a7849b` — "Fix landing page gap: add Fundamental Analysis to
pipeline and concept cards."

### How to Prevent This in Future Projects

The recommended approach is **brief each agent on what the others are building**
when writing the worktree prompts:

```
"Start 3 worktrees:
1. Redesign the landing page. NOTE: the app will have 6 tabs
   including a new Fundamentals tab being built by another agent.
   Include it in your pipeline and concept cards.
2. Add fundamental analysis module and Tab 6.
3. Rewrite docs. NOTE: include the Fundamentals tab in all sections."
```

This costs nothing (a few extra lines in the prompt) and prevents most
cross-feature gaps. Follow up with a post-merge integration check to catch
anything that slipped through.

For a detailed analysis of this problem with four mitigation strategies,
see the "Cross-Feature Awareness Problem" section in
[WORKTREES.md](WORKTREES.md).

---

## Related Documentation

- [WORKTREES.md](WORKTREES.md) — Complete guide to git worktrees in Claude Code:
  what they are, how to invoke them, how to test/merge/clean up
- [MCP_CHROME_EXPLAINED.md](MCP_CHROME_EXPLAINED.md) — Why Chrome MCP commands
  can route to the wrong machine when multiple computers are involved
- [BROWSER_LAUNCH_BUG.md](BROWSER_LAUNCH_BUG.md) — Why `cmd.exe /c start` fails
  silently from Git Bash but direct `chrome.exe` invocation works
