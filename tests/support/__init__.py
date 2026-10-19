"""The fakes, fixtures and helpers that tests of more than one area share.

``tests/conftest.py`` registers every fixture here for the whole suite. Import the other names from
their modules:

- ``tools``: the fake ``podman``, ``btrfs`` and ``findmnt`` on PATH (``fake_podman``, ``fake_btrfs``,
  ``fake_findmnt``, ``fake_btrfs_store``), their knobs and their records, and ``fake_runtime_bin``.
  ``fakes/`` holds the fake executables. Each module docstring there says what the fake answers.
- ``live``: the checks of the large tests that use the real podman and btrfs store
  (``live_podman_store``, ``live_podman_problem``, ``podman_store_in_use``).
- ``runstore``: a Run Store on the PostgreSQL test database, in a new schema for each test
  (``run_store``, ``fresh_run_store``).
- ``process``: the autouse fixture that gives back the umask and the environment after each test.
- ``http``: an ASGI app on a local port (``serving``), a scripted OpenAI chat endpoint
  (``chat_server``, ``chat_completion``), and an endpoint that never answers (``silent_endpoint``).
- ``waiting``: ``wait_until``, which waits for a condition with a deadline instead of a sleep.
- ``paths``: the repository root, ``tests/fixtures`` and the Harbor task ``HARBOR_SHELL_TASK``.

A fake has a fidelity test that runs the same code against the real tool, marked ``slow``.
"""
