# Work done in AgentFoundry (AF)

Gopniya started in Claude Code (commits up to `1a69838` on `main`). On 4 November 2026 we opened
the repository in the AgentFoundry desktop IDE (Linux AppImage, version 0.1.80, signed in with the
team's AgentFoundry account) and built the next feature there, on the branch `agentfoundry`.

## What was built in AF

Three more Indian identifier types, end to end (detector, guard re-scan, placeholders, tests):

| Type | Rule |
|---|---|
| `PASSPORT` | one letter + 7 digits, only after a `Passport` / `पासपोर्ट` label |
| `DRIVING_LICENCE` | state/UT code + 2-digit RTO + 11 digits, with or without separators; labelled (`DL`, `Driving Licence`, `ड्राइविंग लाइसेंस`) or unlabelled when the prefix is a real state/UT code |
| `PMJAY` | ID of 10+ letters/digits with at least 4 digits, after `PM-JAY` / `PMJAY` / `Ayushman` / `आयुष्मान` / `पीएम-जय` |

Files: `gopniya/detect.py`, `gopniya/guard.py`, `gopniya/validators.py`, `tests/test_gopniya.py`
(27 → 34 tests).

## How the work went

1. **SMART-Code agent (AF Cloud, Haiku + GPT routing)** got the task in the SMART-CODE panel. It read
   `detect.py`, `guard.py`, `validators.py`, `vault.py` and the tests, then made the edits and wrote
   the tests. Every tool call went through AF's Confirm mode; the log of 26 approved calls is in
   `agentfoundry-proof/approvals.log`, the full agent transcript in
   `agentfoundry-proof/smart-code-cloud-session.json`.
2. **AF terminal**: pytest showed 30 passed, 1 failed (a licence written `DL-04…` with a dash after
   the state code). The failure went back to SMART-Code, which fixed the regex and validator
   (screenshot `11-af-terminal-pytest-fail.png`).
3. **Evaluation caught a regression.** Running `eval/run_eval.py` on the held-out set showed 207 new
   false alarms: the guard's unlabelled passport pattern matched vaccine kit lot numbers
   (`Kit Lot B7894686`). Precision fell from 98.6% to 88.6%. SMART-Code was told to require a label in
   the guard, to check DL prefixes against the list of state/UT codes, and to require digits in PM-JAY
   IDs. It added the state-code list and the PM-JAY digit check.
4. The free Cloud quota (1M input tokens a day) ran out mid-task (`12-cloud-quota-reached.png`).
   AF Local mode with our Ollama model (qwen3 30B) answered in chat but has no file tools, and its
   suggested code did not match the codebase (`15-local-chat-only.png`,
   `smart-code-local-session.json`). The last step, switching the guard loop to the labelled passport
   pattern and adding three regression tests, was done by hand.
5. Final check in the AF terminal: 34 passed; held-out precision back to 98.6%, recall unchanged
   at 95.7% (`16-af-terminal-tests-eval-pass.png`).

## Evaluation before and after (rules only)

| Set | Recall main → branch | Precision main → branch |
|---|---|---|
| dev (150 notes) | 100% → 100% | 100% → 100% |
| held-out (200 notes) | 95.7% → 95.7% | 98.6% → 98.6% |
| blind 2 (150 notes) | 69.9% → 69.9% | 90.0% → 90.0% |

None of the three evaluation sets contains a labelled passport or driving licence number, so these
numbers show the new rules add no false alarms; they do not measure recall on the new types. The
blind set has PM-JAY IDs of 9 characters; we did not tune the rule to that set, to keep it blind.
