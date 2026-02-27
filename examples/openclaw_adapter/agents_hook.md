# OpenClaw / CrabPath adapter hook

OPENCLAW_STATE=~/.crabpath/main/state.json
OPENCLAW_GRAPH=~/.crabpath/main/graph.json

## Query
python3 examples/openclaw_adapter/query_brain.py ~/.crabpath/main/state.json "<query>" --json

## Learn
crabpath learn --graph ~/.crabpath/main/graph.json --outcome N --fired-ids a,b,c

## Rebuild
python3 examples/openclaw_adapter/init_agent_brain.py ~/.openclaw/workspace ~/.openclaw/agents/main/sessions ~/.crabpath/main

## Health
crabpath health --graph ~/.crabpath/main/graph.json
