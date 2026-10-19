"""The evaluation of a Harbor task with a method, and the gate of a task set, as Run Store clients.

``evaluate`` runs one task: the parent rollout, then the branches that a method (baseline,
sprout, bpo or shepherd) makes from the parent's recovery points. ``policies`` holds the BPO
and Shepherd branch selectors. ``gate`` checks that a task's oracle scores 1 and that doing
nothing scores 0. ``jobs`` builds the Run Store jobs from the task, and ``groups`` keeps the
groups of jobs that the search coordinator runs. ``harbor`` is Harbor's own environment on a
podman sandbox: Harbor's trials on it are the reference for the worker's ``harbor`` grade.
``docs/EVALUATION.md`` describes the options and the output.
"""
