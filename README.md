# MAize Purple Agent

A purple A2A agent for the AgentBeats `MAizeBargAIn` benchmark.

This agent is built for repeated bargaining games where the challenger must negotiate item allocations against multiple opponent strategies and return valid machine-readable actions to the green evaluator.

## Overview

The agent follows a hybrid design:

- deterministic rules for `ACCEPT_OR_REJECT`
- GPT-assisted proposal generation for `PROPOSE`
- catalog-aware proposal selection via `choice_id` when the green agent provides valid options
- strict JSON validation and a rule-based fallback when the model response is invalid

The main goal of this design is not only to negotiate well, but also to stay robust inside the AgentBeats evaluation pipeline.

## Implemented Strategy

### Core idea

The agent treats negotiation as a constrained decision problem:

- never accept offers that are clearly below a useful threshold
- prefer keeping higher-value items
- make controlled concessions over time
- always return valid JSON actions to the evaluator

### Decision flow

```text
Incoming A2A Message
        |
        v
parse_observation()
        |
        v
  +---------------------------+
  | action == ACCEPT_OR_REJECT? |
  +---------------------------+
      | yes                         | no
      v                             v
deterministic threshold check   GPT proposal generation
      |                             |
      v                             v
 ACCEPT / WALK              validate JSON + constraints
                                     |
                        +------------+------------+
                        | valid                    | invalid
                        v                          v
                return allocation JSON     rule-based fallback
```

### Negotiation policy

| Situation | Policy |
|---|---|
| Incoming offer during `ACCEPT_OR_REJECT` | Compute own utility and compare against a round-aware threshold |
| Early rounds | Hold a firmer position and require better value |
| Middle rounds | Relax threshold moderately |
| Late rounds | Allow more compromise if the offer is still above a discounted threshold |
| `PROPOSE` turn with allocation catalog | Score valid catalog entries, then ask GPT to select a strong `choice_id` |
| `PROPOSE` turn without catalog | Ask GPT for a strict JSON allocation proposal |
| `PROPOSE` turn with invalid/missing LLM output | Use the best scored catalog option or a deterministic allocation heuristic |

### Deterministic safeguards

| Safeguard | Purpose |
|---|---|
| JSON-only response handling | Prevent `non-JSON response` failures in the green agent |
| Allocation sum checks | Ensure `allocation_self[i] + allocation_other[i] == quantities[i]` |
| Non-negative allocation checks | Avoid invalid proposals |
| BATNA check | Avoid returning proposals below our own minimum utility threshold |
| Rule-based fallback | Keep the agent alive even when the LLM fails |

### Current strategic behavior

The current version already implements:

- BATNA-aware acceptance logic
- round-dependent concession behavior
- preference for keeping higher-value items
- LLM-assisted proposal generation
- support for green-agent allocation catalogs and `choice_id`
- deterministic fallback proposals
- per-game offer history storage
- penalties against taking back previous concessions too abruptly

The current version does not yet fully implement:

- explicit opponent-type classification (`soft`, `tough`, `aspiration`, `walk`, `nfsp`, `rnad`)
- persuasion-specific response shaping
- anti-persuasion defenses based on detected manipulation patterns
- learned concession schedules

### Mistake avoidance

The green-agent tutorial highlights five common LLM bargaining mistakes. The current agent already addresses most of them at the protocol or scoring layer:

| Mistake | Status | Current handling |
|---|---|---|
| `M1`: make an offer worse than your previous offer | Partial | catalog scoring now penalizes taking back previous concessions |
| `M2`: make an offer worse for you than BATNA | Covered | proposal validation rejects offers below BATNA |
| `M3`: offer no items or all items | Partial | fallback avoids extreme divisions and catalog path prefers balanced valid entries |
| `M4`: accept an offer worse than BATNA | Covered | deterministic `ACCEPT_OR_REJECT` uses a round-aware threshold |
| `M5`: walk away from an offer better than BATNA | Partial | late-round threshold softens, but this can still be improved |

## Architecture

### High-level components

| File | Responsibility |
|---|---|
| `src/agent.py` | Negotiation policy, observation parsing, LLM calls, fallback logic |
| `src/executor.py` | A2A task execution and agent lifecycle per context |
| `src/server.py` | Agent card and HTTP server bootstrap |
| `src/messenger.py` | Helper utilities for A2A-to-A2A communication |
| `amber-manifest.json5` | Deployment manifest for AgentBeats |

### Runtime scheme

```text
AgentBeats Green Agent
        |
        v
    A2A Gateway
        |
        v
  MAize Purple Agent
        |
        +--> parse observation
        +--> choose action type
        +--> deterministic accept/reject
        +--> GPT proposal or fallback proposal
        +--> return JSON artifact
```

### Memory model

The agent keeps per-game offer history in memory while the current container is alive:

- previous offers we made to the opponent
- previous allocations we kept for ourselves

This memory is available across turns within the same assessment run and is keyed by `game_index`.

It is not persistent across fresh container starts, which is exactly what we want for AgentBeats reproducibility: every assessment should begin from a clean state.

## Response formats

### Accept or reject

```json
{"action": "ACCEPT"}
```

```json
{"action": "WALK"}
```

### Proposal

```json
{
  "allocation_self": [3, 2, 1],
  "allocation_other": [0, 1, 2]
}
```

or, when the green agent provides a catalog of valid allocations:

```json
{"choice_id": 4}
```

Where:

- `allocation_self[i] + allocation_other[i] = quantities[i]`
- all values must be integers
- all values must be non-negative
- `choice_id` refers to an entry in the allocation catalog sent by the green evaluator
- in the local MAize green repo these catalog entries are keyed by `id`, and our agent maps that correctly

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | No, but required for GPT proposals | OpenAI key for the purple agent |
| `LLM_MODEL` | Optional | Model name, defaults to `gpt-5-mini` |

If no valid OpenAI key is provided, the agent still works using deterministic fallback proposals.

## Challenger Circle

`challenger_circle` is a green-agent config parameter from the MAize evaluator, not an environment variable inside this repo.

It controls how much structured bargaining guidance the green agent injects into observations sent to remote LLM agents:

| Circle | What gets added |
|---|---|
| 0 | Bare rules only |
| 1 | Objective |
| 2 | Worked example |
| 3 | Step-by-step routine |
| 4 | Common mistakes to avoid |
| 5 | Quick numeric checks |
| 6 | Strategic inference from the opponent's offers |

For our agent, `challenger_circle = 6` is the best default because it gives the model the richest structured prompt without changing our own code.

## Opponent Strategies In The Benchmark

The challenger is evaluated against a baseline pool managed by the green agent:

- `soft`: always-accept style
- `tough`: minimal-offer style
- `aspiration`: concession-schedule heuristic
- `walk`: BATNA-preferring heuristic
- `nfsp`: Neural Fictitious Self-Play
- `rnad`: Regularized Nash Dynamics

The green agent does not directly reveal the opponent label to us during play, so our current policy reacts to observed offers and round structure rather than to the strategy name itself.

## Project Structure

```text
src/
  agent.py
  executor.py
  messenger.py
  server.py
tests/
  conftest.py
  test_agent.py
CHECKLIST.md
amber-manifest.json5
Dockerfile
pyproject.toml
```

## Local Run

### Docker

```powershell
docker build -t my-agent .
docker run --rm -p 9009:9009 my-agent --host 0.0.0.0 --port 9009
```

Agent card:

```text
http://localhost:9009/.well-known/agent-card.json
```

### Local development

```powershell
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```

### Local evaluation against the green agent

The repository `tutorial-agent-beats-comp` contains the local MAize evaluator loop. A practical workflow is:

```powershell
# Terminal 1: green agent
cd tutorial-agent-beats-comp
$env:PYTHONPATH = "scenarios/bargaining/open_spiel"
uv run python -m scenarios.bargaining.bargaining_green serve --port 9029
```

```powershell
# Terminal 2: purple agent
uv run python src/server.py --host 0.0.0.0 --port 9009
```

```powershell
# Terminal 3: one-shot local evaluation
cd tutorial-agent-beats-comp
$env:PYTHONPATH = "scenarios/bargaining/open_spiel"
uv run python -m scenarios.bargaining.bargaining_green once --config (Get-Content ..\local-test-config.example.json -Raw)
```

The included [local-test-config.example.json](/c:/Users/recroot/Documents/katya/ITMO_hub/4%20semestr/j13_agentbeats-purple-agent/local-test-config.example.json) uses:

- `challenger_circle: 6`
- `model: "nfsp"`
- `full_matrix: false`

That makes local checks much faster than a full AgentBeats submit while still exercising the real bargaining protocol.

## AgentBeats Submission Notes

Before submission:

1. Push this repository to a public GitHub repository.
2. Publish a public Docker image.
3. Ensure the `amber-manifest.json5` image reference is correct.
4. Register the purple agent on `agentbeats.dev`.
5. During `Quick Submit`, provide:
   - green-agent model config and key
   - purple-agent key if you want GPT-backed proposal generation

## Known Design Choices

- The agent is optimized first for robustness and valid interaction with the benchmark.
- Deterministic logic is used where the action boundary is clear.
- LLM usage is concentrated in proposal generation, where flexible search is more useful.
- When a catalog of valid allocations is available, the agent prefers choosing among valid options instead of inventing a split from scratch.
- Fallback logic is intentionally simple so the agent can still complete games even if the model output is malformed.

## Next Improvements

- opponent-style classification
- anti-persuasion heuristics
- better use of offer history
- stronger proposal prompts
- explicit Nash-style proposal targets
