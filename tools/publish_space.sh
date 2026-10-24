#!/bin/sh
# Publishes the committed code to a Hugging Face Space (a Docker Space).
# Usage: tools/publish_space.sh <hf-username>/<space-name>
# The Space repo needs its own README header, so the copy is built in a temporary folder and the
# README in this repository is left alone. Git asks for your Hugging Face username and, as the password,
# an access token with write permission (huggingface.co/settings/tokens).
set -eu
SPACE="${1:?usage: tools/publish_space.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git -C "$ROOT" archive HEAD | tar -x -C "$TMP"
{
  printf -- '---\ntitle: USC Pre-PT Club Movement Analysis\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n'
  cat "$TMP/README.md"
} > "$TMP/README.new"
mv "$TMP/README.new" "$TMP/README.md"

cd "$TMP"
git init -q -b main
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-$(git -C "$ROOT" config user.name)}" \
    -c user.email="${GIT_AUTHOR_EMAIL:-$(git -C "$ROOT" log -1 --format=%ae)}" \
    commit -q -m "Deploy from GitHub $(git -C "$ROOT" rev-parse --short HEAD)"
git push --force "https://huggingface.co/spaces/$SPACE" main
echo "Pushed. Build progress: https://huggingface.co/spaces/$SPACE"
