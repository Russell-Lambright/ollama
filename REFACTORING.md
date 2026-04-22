# Major Refactor / Enhancement Branch

## Branch: `feature/major-refactor`

This branch (`copilot/featuremajor-refactor`) was created off `main` to isolate work for a major refactor or enhancement of the Ollama project.

## Purpose

Use this branch to:

- Introduce large-scale architectural changes without disrupting the stability of `main`.
- Develop and test enhancements iteratively before merging back.
- Keep a clean history of refactor-related commits separate from day-to-day fixes.

## Naming Convention

Branches for significant features or refactors should follow:

```
feature/<short-description>
refactor/<short-description>
enhancement/<short-description>
```

Examples:
- `feature/major-refactor`
- `refactor/model-loading`
- `enhancement/streaming-api`

## Workflow

1. Branch off the latest `main`:
   ```bash
   git fetch origin
   git checkout -b feature/major-refactor origin/main
   ```

2. Push to remote to track changes:
   ```bash
   git push -u origin feature/major-refactor
   ```

3. Open a pull request against `main` when the work is ready for review.

## Guidelines

- Keep commits focused and well-described.
- Rebase on `main` regularly to reduce merge conflicts.
- Add or update tests for all changed code paths.
- Update documentation alongside code changes.
