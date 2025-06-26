# Contributing Guide

Thanks for considering contributing to this project! To keep things clean and maintainable, please follow the guidelines below.

## commit Message Convention

We use the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>(<scope>): <summary>

<body>
```

- `type`: The category of the change (see below)
- `scope`: The part of the codebase the change affects (optional but recommended)
- `summary`: A short, imperative sentence describing the change
- `body`: (optional) A detailed explanation of the change and its reasoning

### Common Types

| Type     | Description                                         |
| -------- | --------------------------------------------------- |
| feat     | A new feature                                       |
| fix      | A bug fix                                           |
| docs     | Documentation changes only                          |
| style    | Code style changes (formatting, etc.)               |
| refactor | Code changes that neither fix bugs nor add features |
| test     | Adding or updating tests                            |
| chore    | Maintenance tasks (build process, tools, etc.)      |
| perf     | Performance improvements                            |
| ci       | Changes to CI/CD configuration                      |
| build    | Changes to build system or dependencies             |
| revert   | Reverting a previous commit                         |

### Example

```
feat(cli): add --dry-run flag to simulate deletion

This adds a new `--dry-run` option to the CLI. When enabled, the command will simulate
file deletions without actually removing them. Useful for debugging large batch runs.

Closes #42
```

## Changelog Policy

Generate changelog.md file everytime we publish a new release:

- Before **each release** (beta or stable), please update `tools/generate_changelog.sh` and run:

```bash
cd path/to/tidymut
bash tools/generate_changelog.sh
```

To avoid cluttering the `main` branch with release artifacts,
changelog updates are typically made on a `release/*` branch.
If desired, you can manually merge only the `changelog.md` back
into `main` using git checkout `release/* -- CHANGELOG.md`.

## Tips

- Use `git commit -m "type(scope): summary" -m "body"` for multi-line messages.
- Use `git add -p` to split large commits into atomic ones.
- Use `git tag vX.Y.Z` to prepare for releases.

## Squash Commit Guidelines

When using squash-and-merge (especially via GitHub UI), all individual commits in a feature branch are combined into a single commit. Please follow this format for the final squash commit message:

```text
<type>(<scope>): <summary>

<body>
```

- Follow the same Conventional Commits format.
- The body can include bullet points summarizing the key changes if the branch includes multiple related edits.

### Example

```text
feat(parser): add support for multi-mutation parsing

- Added helper to parse comma-separated mutations
- Integrated fallback validator
- Updated tests for multi-mutation strings
```

You can edit the squash message in the GitHub UI before confirming the merge.

## Full Release Workflow

Here is a complete step-by-step workflow for submitting code from a feature branch, preparing a release, and generating a changelog:

```bash
# >>> Fix bugs or Add feats >>>
# Check your branch before start
git branch
# Develop on feature branch
git checkout -b feature/my-feature
# edit files...
git commit -m "feat(core): ..."
git push origin feature/my-feature

# For Manager: open a pull request to merge the feature branch into main
# After code review and testing, use squash and merge in GitHub UI.
# Or you can do it in command line:
git checkout main
git pull origin main

## squash merge
git merge --squash feature/your-feature-name

## squash commit
git commit -m "See 'Squash Commit Guidelines' for details."
git push origin main

# After squash-and-merge, sync your local main branch
git checkout main
git pull origin main

# >>> The instructions below are for maintainers preparing a release >>>
# Generate changelog when a new version is ready 
## See Changelog Policy for details
bash tool/generate_changelog.sh

## Commit the changelog
git checkout main
git add path/to/changelog.md  # in doc/changelog/
git commit -m "docs(changelog): update main after version release."  # replace with real version

## tag the release
git tag version-tag  # replace with real version
git push origin version-tag  # replace with real version
git tag -l
git show v0.1.0-beta
```

## Resources

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Releases Guide](https://docs.github.com/en/repositories/releasing-projects-on-github)

---

Thank you for contributing! 🙌
