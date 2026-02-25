#!/bin/bash
# CrabPath Quickstart — from zero to shadow mode
# Usage: bash scripts/quickstart.sh ~/.openclaw/workspace
#
# What it does:
#   1. Bootstraps a graph from your workspace files
#   2. Builds embeddings (requires OPENAI_API_KEY or GEMINI_API_KEY)
#   3. Runs a health check
#   4. Runs 5 test queries with scoring
#   5. Shows you what the brain learned

set -e

WORKSPACE="${1:-$HOME/.openclaw/workspace}"
GRAPH="graph.json"
EMBEDDINGS="embeddings.json"
STATE_DIR="$HOME/.crabpath"

echo "🦀 CrabPath Quickstart"
echo "========================"
echo "Workspace: $WORKSPACE"
echo ""

# Check workspace exists
if [ ! -d "$WORKSPACE" ]; then
    echo "❌ Workspace not found: $WORKSPACE"
    echo "Usage: bash scripts/quickstart.sh /path/to/workspace"
    exit 1
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  No OPENAI_API_KEY or GEMINI_API_KEY found."
    echo "   Embeddings will use keyword fallback (less precise)."
    echo "   Set one for better results: export OPENAI_API_KEY=sk-..."
    echo ""
    HAS_KEY=false
else
    HAS_KEY=true
fi

# Step 1: Bootstrap
echo "Step 1/5: Bootstrapping graph from workspace..."
mkdir -p "$STATE_DIR"
python3 -m crabpath.cli migrate \
    --workspace "$WORKSPACE" \
    --include-memory \
    --output-graph "$STATE_DIR/$GRAPH" \
    ${HAS_KEY:+--output-embeddings "$STATE_DIR/$EMBEDDINGS"} \
    --verbose 2>&1 | tail -5
echo "✅ Graph bootstrapped"
echo ""

# Step 2: Health check
echo "Step 2/5: Health check..."
python3 -m crabpath.cli health --graph "$STATE_DIR/$GRAPH" 2>&1
echo ""

# Step 3: Evolution baseline
echo "Step 3/5: Recording evolution baseline..."
python3 -m crabpath.cli evolve \
    --graph "$STATE_DIR/$GRAPH" \
    --snapshots "$STATE_DIR/evolution.jsonl" 2>&1
echo "✅ Baseline recorded"
echo ""

# Step 4: Test queries (if embeddings available)
if [ "$HAS_KEY" = true ] && [ -f "$STATE_DIR/$EMBEDDINGS" ]; then
    echo "Step 4/5: Running 5 test queries..."
    for q in \
        "what are the main rules and guidelines" \
        "how do I use the coding tools" \
        "what are the safety rules" \
        "who am I helping" \
        "what happened recently"; do
        echo "  Query: $q"
        python3 -m crabpath.cli query \
            --graph "$STATE_DIR/$GRAPH" \
            --index "$STATE_DIR/$EMBEDDINGS" \
            --top 3 \
            "$q" 2>&1 | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read())
    for n in d.get('fired_nodes',[])[:3]:
        print(f'    → {n[\"id\"][:40]}: {n[\"content\"][:60]}...')
except: print('    (parse error)')
"
        echo ""
    done
    echo "✅ Test queries complete"
else
    echo "Step 4/5: Skipped (no API key for embeddings/scoring)"
fi
echo ""

# Step 5: Summary
echo "Step 5/5: Summary"
echo "========================"
echo "Graph:      $STATE_DIR/$GRAPH"
echo "Embeddings: $STATE_DIR/$EMBEDDINGS"
echo "Evolution:  $STATE_DIR/evolution.jsonl"
echo "Shadow log: $STATE_DIR/shadow.jsonl"
echo ""
echo "Next steps:"
echo "  1. Add shadow mode to your agent (see PLAYBOOK.md)"
echo "  2. Monitor: python3 -m crabpath.cli health --graph $STATE_DIR/$GRAPH"
echo "  3. Track:   python3 -m crabpath.cli evolve --graph $STATE_DIR/$GRAPH --snapshots $STATE_DIR/evolution.jsonl --report"
echo ""
echo "🦀 CrabPath is ready. The brain will learn from every query."
