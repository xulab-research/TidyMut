#!/bin/bash
# generate_changelog.sh
# Reminder: Update this file each time you commit to GitHub 
# to ensure the changelog and commit history stay consistent.

# Validate gitHub Token
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Please set GITHUB_TOKEN as environment variables"
    exit 1
fi

# Check version range
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -z "$LAST_TAG" ]; then
    FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
    REVISION_RANGE="${FIRST_COMMIT}..HEAD"
    echo "Commit first time: $REVISION_RANGE"
else
    REVISION_RANGE="${LAST_TAG}..HEAD"
    echo "Commit new version, range : $REVISION_RANGE"
fi

# Generate changelog
echo "Generating changelog..."
python tools/changelog.py \
    "$GITHUB_TOKEN" \
    "xulab-research/TidyMut" \
    "$REVISION_RANGE" \
    --template keepachangelog \
    --output ./doc/changelog/CHANGELOG_0.3.0.md

if [ -f "./doc/changelog/CHANGELOG_0.3.0.md" ]; then
    echo "Generate changelog successfully!"
    echo "File: CHANGELOG_0.3.0.md"
else
    echo "Failed to generate changelog."
fi