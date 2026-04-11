# AgentBeats MAizeBargAIn Checklist

## 1. Packaging and Registration

- [ ] Fill in agent metadata in `src/server.py`
- [ ] Fill in `README.md` with agent description and run instructions
- [ ] Replace placeholders in `amber-manifest.json5`
- [ ] Make sure required env vars and secrets are defined clearly
- [ ] Verify the Docker image builds successfully
- [ ] Verify `http://localhost:9009/.well-known/agent-card.json` responds correctly
- [ ] Push the repo to public GitHub
- [ ] Publish a public Docker image
- [ ] Register the purple agent on `agentbeats.dev`

## 2. Baseline Agent Logic

- [ ] Implement baseline negotiation logic in `src/agent.py`
- [ ] Add a lightweight negotiation state model
- [ ] Store short dialogue history
- [ ] Add simple opponent classification
- [ ] Add a concession policy
- [ ] Add a minimum acceptable threshold
- [ ] Make responses short, stable, and consistent
- [ ] Handle malformed or empty input safely

## 3. Local Validation

- [ ] Run tests locally
- [ ] Smoke-test the container locally
- [ ] Check that the agent does not crash on unexpected messages
- [ ] Check that the agent always returns a valid text response
- [ ] Verify logs are understandable for debugging

## 4. First Submission

- [ ] Open the `MAizeBargAIn` green-agent page on `agentbeats.dev`
- [ ] Use `Quick Submit`
- [ ] Select the registered purple agent
- [ ] Add required secrets
- [ ] Submit the run
- [ ] Save the PR link in the leaderboard repo
- [ ] Save the workflow run link
- [ ] Save the resulting score/metrics

## 5. Iteration After First Valid Run

- [ ] Review the PR result and metrics
- [ ] Write down what failed or looked weak
- [ ] Improve state tracking
- [ ] Improve opponent modeling
- [ ] Add anti-persuasion rules
- [ ] Add persuasion and framing tactics
- [ ] Tune concession behavior by opponent type
- [ ] Re-submit and compare results

## 6. Tracking Progress

- [ ] Record each submission date
- [ ] Record what changed before each submission
- [ ] Record PR links and scores
- [ ] Keep the best-performing version easy to identify
