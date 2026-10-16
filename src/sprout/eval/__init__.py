"""What counts as an answer, and what to do when the answer is wrong.

``evaluate`` runs one Harbor task: the parent rollout, then a method (baseline,
sprout, bpo, shepherd) that branches from the parent's snapshots. ``branching``
is the analyst/reviewer machinery sprout uses; ``policies`` the BPO and
Shepherd selectors; ``gate`` checks a dataset's grader through our transport.
"""
