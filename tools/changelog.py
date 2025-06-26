#!/usr/bin/env python3
"""
Enhanced changelog generator with templates and categorization

This script generates structured changelogs with templates and automatic
PR categorization based on labels, titles, and content.

Usage::

    $ python changelog_generator.py <token> <repo> <revision_range> [--template <template>]

Examples::

    $ python changelog_generator.py $GITHUB_TOKEN owner/repo v1.0.0..v1.1.0
    $ python changelog_generator.py $GITHUB_TOKEN owner/repo v1.0.0..v1.1.0 --template keepachangelog

Dependencies
------------

- gitpython
- pygithub
- jinja2
- git >= 2.29.0

Install dependencies:
    pip install gitpython pygithub jinja2
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    from git import Repo
    from github import Github
    from jinja2 import Template
except ImportError:
    print("Missing dependencies. Install with:")
    print("pip install gitpython pygithub jinja2")
    sys.exit(1)

# 可配置的机器人名单
BOTS_TO_IGNORE = {
    "Homu",
    "dependabot-preview",
    "dependabot[bot]",
    "github-actions[bot]",
    "codecov[bot]",
    "pre-commit-ci[bot]",
}

# PR 分类规则
CATEGORY_RULES = {
    "breaking": {
        "keywords": ["breaking", "breaking change", "breaking-change", "bc", "💥"],
        "labels": ["breaking-change", "breaking", "major"],
        "priority": 1,
    },
    "features": {
        "keywords": ["feat", "feature", "add", "new", "✨", "🎉"],
        "labels": ["enhancement", "feature", "new-feature", "minor"],
        "priority": 2,
    },
    "fixes": {
        "keywords": ["fix", "bug", "bugfix", "hotfix", "patch", "🐛", "🚑"],
        "labels": ["bug", "bugfix", "fix", "patch"],
        "priority": 3,
    },
    "security": {
        "keywords": ["security", "vulnerability", "cve", "🔒", "🛡️"],
        "labels": ["security", "vulnerability"],
        "priority": 4,
    },
    "performance": {
        "keywords": ["perf", "performance", "optimize", "speed", "⚡", "🚀"],
        "labels": ["performance", "optimization"],
        "priority": 5,
    },
    "documentation": {
        "keywords": ["doc", "docs", "documentation", "readme", "📚", "📝"],
        "labels": ["documentation", "docs"],
        "priority": 6,
    },
    "tests": {
        "keywords": ["test", "testing", "spec", "coverage", "🧪", "✅"],
        "labels": ["tests", "testing"],
        "priority": 7,
    },
    "dependencies": {
        "keywords": ["dep", "deps", "dependency", "dependencies", "bump", "⬆️", "📦"],
        "labels": ["dependencies", "deps"],
        "priority": 8,
    },
    "ci": {
        "keywords": ["ci", "cd", "build", "workflow", "action", "👷", "🔧"],
        "labels": ["ci", "build", "workflow"],
        "priority": 9,
    },
    "refactor": {
        "keywords": ["refactor", "refactoring", "cleanup", "clean", "♻️", "🎨"],
        "labels": ["refactor", "cleanup"],
        "priority": 10,
    },
}

# 模板定义
TEMPLATES = {
    "keepachangelog": """# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [{{ version }}] - {{ date }}

{% if breaking_changes -%}
### Breaking Changes
{% for pr in breaking_changes %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if features -%}
### Added
{% for pr in features %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if fixes -%}
### Fixed
{% for pr in fixes %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if security -%}
### Security
{% for pr in security %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if performance -%}
### Performance
{% for pr in performance %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if documentation -%}
### Documentation
{% for pr in documentation %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if dependencies -%}
### Dependencies
{% for pr in dependencies %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
{% if other_changes -%}
### Other Changes
{% for pr in other_changes %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }})){% if pr.user.login %} by @{{ pr.user.login }}{% endif %}
{%- endfor %}

{% endif -%}
### Contributors

{{ contributors_message }}

{% for author in authors -%}
- {{ author }}
{% endfor %}

**Full Changelog**: [{{ revision_range }}]({{ compare_url }})
""",
    "github": """## {{ version }} ({{ date }})

{% if breaking_changes -%}
### BREAKING CHANGES
{% for pr in breaking_changes %}
* {{ pr.title }} by @{{ pr.user.login }} in [#{{ pr.number }}]({{ pr.html_url }})
{%- endfor %}

{% endif -%}
{% if features -%}
### Features
{% for pr in features %}
* {{ pr.title }} by @{{ pr.user.login }} in [#{{ pr.number }}]({{ pr.html_url }})
{%- endfor %}

{% endif -%}
{% if fixes -%}
### Bug Fixes
{% for pr in fixes %}
* {{ pr.title }} by @{{ pr.user.login }} in [#{{ pr.number }}]({{ pr.html_url }})
{%- endfor %}

{% endif -%}
{% if performance -%}
### Performance Improvements
{% for pr in performance %}
* {{ pr.title }} by @{{ pr.user.login }} in [#{{ pr.number }}]({{ pr.html_url }})
{%- endfor %}

{% endif -%}
{% if other_changes -%}
### Other Changes
{% for pr in other_changes %}
* {{ pr.title }} by @{{ pr.user.login }} in [#{{ pr.number }}]({{ pr.html_url }})
{%- endfor %}

{% endif -%}
### New Contributors
{% for author in new_authors -%}
* @{{ author }} made their first contribution
{% endfor %}

**Full Changelog**: {{ compare_url }}
""",
    "simple": """# Release {{ version }} ({{ date }})

{% if features -%}
## New Features
{% for pr in features %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }}))
{%- endfor %}

{% endif -%}
{% if fixes -%}
## Bug Fixes
{% for pr in fixes %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }}))
{%- endfor %}

{% endif -%}
{% if other_changes -%}
## Other Changes
{% for pr in other_changes %}
- {{ pr.title }} ([#{{ pr.number }}]({{ pr.html_url }}))
{%- endfor %}

{% endif -%}
## Contributors ({{ total_contributors }})

Thank you to everyone who contributed to this release!

{% for author in authors -%}
- {{ author }}
{% endfor %}
""",
    "rst": """{{ version }} ({{ date }})
{{ "=" * (version|length + date|length + 3) }}

{% if breaking_changes -%}
Breaking Changes
----------------
{% for pr in breaking_changes %}
* {{ pr.title }} (`#{{ pr.number }} <{{ pr.html_url }}>`__)
{%- endfor %}

{% endif -%}
{% if features -%}
New Features
------------
{% for pr in features %}
* {{ pr.title }} (`#{{ pr.number }} <{{ pr.html_url }}>`__)
{%- endfor %}

{% endif -%}
{% if fixes -%}
Bug Fixes
---------
{% for pr in fixes %}
* {{ pr.title }} (`#{{ pr.number }} <{{ pr.html_url }}>`__)
{%- endfor %}

{% endif -%}
{% if other_changes -%}
Other Changes
-------------
{% for pr in other_changes %}
* {{ pr.title }} (`#{{ pr.number }} <{{ pr.html_url }}>`__)
{%- endfor %}

{% endif -%}
Contributors
------------

{{ contributors_message }}

{% for author in authors -%}
* {{ author }}
{% endfor %}
""",
}


def find_git_repo():
    """Find the git repository root from current directory or script location"""
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / ".git").exists():
            return Repo(str(parent))

    script_dir = Path(__file__).parent
    for parent in [script_dir] + list(script_dir.parents):
        if (parent / ".git").exists():
            return Repo(str(parent))

    raise ValueError("Could not find git repository. Run from within a git repo.")


def get_authors(repo, revision_range):
    """Extract authors from git history"""
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]
    authors_pat = r"^.*\t(.*)$"

    grp1 = "--group=author"
    grp2 = "--group=trailer:co-authored-by"

    try:
        cur = repo.git.shortlog("-s", grp1, grp2, revision_range)
        pre = repo.git.shortlog("-s", grp1, grp2, lst_release)
    except Exception as e:
        print(f"Error getting git shortlog: {e}")
        sys.exit(1)

    authors_cur = set(re.findall(authors_pat, cur, re.M))
    authors_pre = set(re.findall(authors_pat, pre, re.M))

    # Remove bots
    for bot in BOTS_TO_IGNORE:
        authors_cur.discard(bot)
        authors_pre.discard(bot)

    # Separate new and existing authors
    authors_new = list(authors_cur - authors_pre)
    authors_old = list(authors_cur & authors_pre)

    # Mark new authors with +
    authors_new_marked = [s + " +" for s in authors_new]
    all_authors = authors_new_marked + authors_old
    all_authors.sort()

    return {
        "all": all_authors,
        "new": authors_new,
        "existing": authors_old,
        "total": len(authors_cur),
    }


def categorize_pr(pr):
    """Categorize a PR based on its title and labels"""
    title_lower = pr.title.lower()
    pr_labels = [label.name.lower() for label in pr.labels]

    # Check each category
    for category, rules in CATEGORY_RULES.items():
        # Check keywords in title
        if any(keyword in title_lower for keyword in rules["keywords"]):
            return category

        # Check labels
        if any(label in pr_labels for label in rules["labels"]):
            return category

    return "other"


def get_pull_requests(repo, github_repo, revision_range):
    """Extract and categorize pull requests from git history"""
    prnums = set()

    try:
        # From regular merges
        merges = repo.git.log("--oneline", "--merges", revision_range)

        patterns = [
            r"Merge pull request \#(\d+)",
            r"Auto merge of \#(\d+)",
            r"Merged PR (\d+):",
        ]

        for pattern in patterns:
            issues = re.findall(pattern, merges)
            prnums.update(int(s) for s in issues)

        # From squash merges
        commits = repo.git.log(
            "--oneline", "--no-merges", "--first-parent", revision_range
        )

        pr_patterns = [
            r"^.*\((\#|gh-|gh-\#)(\d+)\)$",
            r"^.*\(\#(\d+)\)$",
            r".*\(#(\d+)\).*",
        ]

        for pattern in pr_patterns:
            matches = re.findall(pattern, commits, re.M)
            if matches and isinstance(matches[0], tuple):
                prnums.update(int(match[-1]) for match in matches)
            else:
                prnums.update(int(match) for match in matches)

    except Exception as e:
        print(f"Warning: Error extracting PR numbers: {e}")

    # Fetch PRs and categorize
    prnums = sorted(list(prnums))
    categorized_prs = defaultdict(list)

    print(f"Found {len(prnums)} pull request references")

    for n in prnums:
        try:
            pr = github_repo.get_pull(n)
            category = categorize_pr(pr)
            categorized_prs[category].append(pr)
        except Exception as e:
            print(f"Warning: Could not fetch PR #{n}: {e}")

    return categorized_prs


def generate_changelog(
    template_name, version, revision_range, repo_name, authors, categorized_prs
):
    """Generate changelog using specified template"""
    if template_name not in TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}"
        )

    template = Template(TEMPLATES[template_name])

    # Prepare template variables
    context = {
        "version": version,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "revision_range": revision_range,
        "compare_url": f"https://github.com/{repo_name}/compare/{revision_range}",
        "authors": authors["all"],
        "new_authors": [
            author.replace(" +", "")
            for author in authors["all"]
            if author.endswith(" +")
        ],
        "total_contributors": authors["total"],
        "contributors_message": f"A total of {authors['total']} people contributed to this release. People with a \"+\" by their names contributed a patch for the first time.",
        # Categorized PRs
        "breaking_changes": categorized_prs.get("breaking", []),
        "features": categorized_prs.get("features", []),
        "fixes": categorized_prs.get("fixes", []),
        "security": categorized_prs.get("security", []),
        "performance": categorized_prs.get("performance", []),
        "documentation": categorized_prs.get("documentation", []),
        "tests": categorized_prs.get("tests", []),
        "dependencies": categorized_prs.get("dependencies", []),
        "ci": categorized_prs.get("ci", []),
        "refactor": categorized_prs.get("refactor", []),
        "other_changes": categorized_prs.get("other", []),
    }

    return template.render(**context)


def main(
    token, repo_name, revision_range, template_name="keepachangelog", output_file=None
):
    """Generate changelog for the specified repository and revision range"""
    try:
        # Initialize git repo
        git_repo = find_git_repo()
        print(f"Using git repository: {git_repo.working_dir}")

        # Initialize GitHub API
        github = Github(token)
        github_repo = github.get_repo(repo_name)
        print(f"Using GitHub repository: {repo_name}")

    except Exception as e:
        print(f"Error initializing: {e}")
        sys.exit(1)

    # Validate revision range
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]
    try:
        git_repo.git.rev_parse(lst_release)
        git_repo.git.rev_parse(cur_release)
    except Exception as e:
        print(f"Invalid revision range {revision_range}: {e}")
        sys.exit(1)

    # Extract version from current release tag
    version = cur_release.lstrip("v")

    print(f"Generating changelog for {revision_range} using template '{template_name}'")
    print("=" * 50)

    # Get authors and PRs
    authors = get_authors(git_repo, revision_range)
    categorized_prs = get_pull_requests(git_repo, github_repo, revision_range)

    # Generate changelog
    changelog = generate_changelog(
        template_name, version, revision_range, repo_name, authors, categorized_prs
    )

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(changelog)
        print(f"Changelog written to {output_file}")
    else:
        print(changelog)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate structured changelog with templates")
    parser.add_argument("token", help="GitHub access token")
    parser.add_argument("repo", help="GitHub repository (owner/repo)")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    parser.add_argument(
        "--template",
        "-t",
        default="keepachangelog",
        choices=list(TEMPLATES.keys()),
        help="Changelog template to use",
    )
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    args = parser.parse_args()
    main(args.token, args.repo, args.revision_range, args.template, args.output)
