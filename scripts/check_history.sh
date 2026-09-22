#!/bin/sh
# Refuse credentials, key material, account ids, resource names, e-mail
# addresses and local paths in the working tree and in every commit that is
# not yet on the base branch. Exit 1 on any hit.
#
# Usage: scripts/check_history.sh [base-ref]   (default origin/main)
# The base ref is scanned only for the tree at HEAD, not its history: lines
# already on the base were reviewed when they merged, and a public history
# cannot be rewritten by a script.
set -eu
cd "$(git rev-parse --show-toplevel)"
BASE="${1:-origin/main}"
PATTERNS='AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|net|org|dev|io)|\b[0-9]{12}\b|rawlogsbucket[0-9a-f]{6,}-|pra-(dev|prod)-[a-z]|arn:aws:(iam|sqs|dynamodb):|sqs\.[a-z0-9-]+\.amazonaws\.com|/Users/[a-z]|/home/[a-z]+/'
ALLOW='example\.com|arn:aws:s3:::\{BUCKET\}|op://'
tree_hits=$(git grep -nIiE "$PATTERNS" -- . ':!uv.lock' ':!scripts/check_history.sh' | grep -vE "$ALLOW" || true)
new_hits=$(git log "$BASE..HEAD" -p --format='COMMIT %h' -- . ':!uv.lock' ':!scripts/check_history.sh' 2>/dev/null \
  | grep -E "^COMMIT|^\+" | grep -v '^+++' | grep -iE "$PATTERNS" | grep -vE "$ALLOW" || true)
status=0
if [ -n "$tree_hits" ]; then echo "check_history: hits in the working tree:" >&2; echo "$tree_hits" | head -40 >&2; status=1; fi
if [ -n "$new_hits" ]; then echo "check_history: hits in commits not on $BASE:" >&2; echo "$new_hits" | head -40 >&2; status=1; fi
[ $status -eq 0 ] && echo "check_history: clean (tree and $BASE..HEAD)"
exit $status
