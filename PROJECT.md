# Project Documentation

This document contains project-level information, guidelines, and templates for the InvokeAI `qt6-migration` branch.

## Active Migration Work

- Branch: `qt6-migration`
- Focus: Qt6 frontend migration
- Local GUI mode: PySide6-based desktop picker with "Local" and "Server" options
- macOS/darwin support: removed from classifiers, dependency markers, uv environments, backend checks, docs, and code comments
- Minimum platform versions: Windows 11+, Ubuntu 24.04+, Arch Linux rolling
- Python versions: 3.13 stable; 3.14/3.15 marked WIP pending Qt6 validation
- Key dependencies bumped: `onnx>=1.23.0`, `numpy` unpinned for 3.14 compatibility
- Key new dependency: `PySide6` added for local GUI mode

## Pull Request Template

### Summary

<!--A description of the changes in this PR. Include the kind of change (fix, feature, docs, etc), the "why" and the "how". Screenshots or videos are useful for frontend changes.-->

### Related Issues / Discussions

<!--WHEN APPLICABLE: List any related issues or discussions on github or discord. If this PR closes an issue, please use the "Closes #1234" format, so that the issue will be automatically closed when the PR merges.-->

### QA Instructions

<!--WHEN APPLICABLE: Describe how you have tested the changes in this PR. Provide enough detail that a reviewer can reproduce your tests.-->

### Merge Plan

<!--WHEN APPLICABLE: Large PRs, or PRs that touch sensitive things like DB schemas, may need some care when merging. For example, a careful rebase by the change author, timing to not interfere with a pending release, or a message to contributors on discord after merging.-->

### Checklist

- [ ] _The PR has a short but descriptive title, suitable for a changelog_
- [ ] _Tests added / updated (if applicable)_
- [ ] _❗Changes to a redux slice have a corresponding migration_
- [ ] _Documentation added / updated (if applicable)_
- [ ] _Updated `What's New` copy (if doing a release after this PR)_
