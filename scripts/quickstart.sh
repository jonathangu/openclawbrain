#!/bin/bash
# CrabPath Quickstart — from zero to shadow mode
# Usage: bash scripts/quickstart.sh [workspace_path]
#
# What it does:
#   1. Installs CrabPath with embedding support (pip install -e .[embeddings])
#   2. Bootstraps a graph from your workspace files
#   3. Builds embeddings (requires OPENAI_API_KEY or GEMINI_API_KEY)
#   4. Runs a health check
#   5. Runs 5 test queries with scoring
#   6. Shows you what the brain learned

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
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

# Step 0: Install CrabPath
echo "Step 0: Installing CrabPath..."
if ! python3 -c "import crabpath" 2>/dev/null; then
    pip3 install -e "$REPO_DIR[embeddings]" --quiet 2>&1 | tail -2
    echo "✅ CrabPath installed with embedding support"
else
    echo "✅ CrabPath already installed"
fi
echo ""

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
        crabpath query \
            --graph "$STATE_DIR/$GRAPH" \
            --index "$STATE_DIR/$EMBEDDINGS" \
            --top 3 \
            "$q" 2>&1 | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read())
    for n in d.get('fired',d.get('fired_nodes',[]))[:3]:
        nid=n['id'][:45]; txt=n['content'][:60].replace(chr(10),' ')
        print(f'    → {nid}: {txt}...')
except Exception as e: print(f'    (parse error: {e})')
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
