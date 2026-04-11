# MAize Purple Agent

A purple A2A agent for the AgentBeats `MAizeBargAIn` negotiation benchmark.

## Status

This repository is based on the official `agent-template` and currently provides:

- an A2A-compatible server
- a Dockerized runtime
- a baseline negotiation-agent scaffold for further strategy work

The main agent logic lives in `src/agent.py`.

## Project Structure

```text
src/
  agent.py       # negotiation logic
  executor.py    # A2A request handling
  messenger.py   # A2A messaging helpers
  server.py      # agent card and server startup
tests/
  test_agent.py  # basic A2A conformance tests
```

## Run Locally

```powershell
docker build -t my-agent .
docker run --rm -p 9009:9009 my-agent --host 0.0.0.0 --port 9009
```

Agent card:

```text
http://localhost:9009/.well-known/agent-card.json
```

## Development

```powershell
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```

## Registration Notes

Before registering this agent on AgentBeats:

- publish the repository publicly on GitHub
- publish a public Docker image
- replace the placeholder image in `amber-manifest.json5`
- provide any required API keys as AgentBeats secrets during submit

## Goal

The goal is to evolve this baseline into a stronger negotiation agent with:

- dialogue state tracking
- opponent modeling
- controlled concessions
- persuasion and anti-persuasion tactics
