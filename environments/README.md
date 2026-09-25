# Environment specifications

Formal benchmark releases will commit rebuildable specifications for two Conda
environments:

```text
env_pysr = matched upstream PySR foundation
env_mysr = MySR development and release environment
```

Environment names are human-facing roles, not sufficient provenance. Each result
must also record exact Python, Julia, package, operating-system, hardware, and Git
versions. Generated cross-platform lock files may be added after the package
management strategy is selected and verified.

The local environments themselves must never be committed.
