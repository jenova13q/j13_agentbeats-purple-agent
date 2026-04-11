# MAize Purple Agent

A purple A2A agent for the AgentBeats `MAizeBargAIn` benchmark.

This agent is built for repeated bargaining games where the challenger must negotiate item allocations against multiple opponent strategies and return valid machine-readable actions to the green evaluator.

## Overview

The agent follows a hybrid design:

- deterministic rules for `ACCEPT_OR_REJECT`
- GPT-assisted proposal generation for `PROPOSE`
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
| `PROPOSE` turn with working LLM | Ask GPT for a strict JSON allocation proposal |
| `PROPOSE` turn with invalid/missing LLM output | Use a deterministic allocation heuristic |

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
- deterministic fallback proposals
- per-game offer history storage for future improvements

The current version does not yet fully implement:

- explicit opponent modeling
- persuasion-specific response shaping
- anti-persuasion defenses based on detected manipulation patterns
- learned concession schedules

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

Where:

- `allocation_self[i] + allocation_other[i] = quantities[i]`
- all values must be integers
- all values must be non-negative

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | No, but required for GPT proposals | OpenAI key for the purple agent |
| `OPENROUTER_API_KEY` | Optional alternative | OpenRouter key for OpenAI-compatible calls |
| `MODEL` | Optional | Model name, defaults to `gpt-4.1-mini` |

If no valid LLM key is provided, the agent still works using deterministic fallback proposals.

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
- Fallback logic is intentionally simple so the agent can still complete games even if the model output is malformed.

## Next Improvements

- opponent-style classification
- anti-persuasion heuristics
- better use of offer history
- stronger proposal prompts
- explicit Nash-style proposal targets
