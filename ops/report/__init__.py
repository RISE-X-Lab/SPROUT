"""Read-only inspection of what a run leaves on disk.

``python -m ops.report`` reads journals, receipts, summaries and plans, and
prints what they say. It writes nothing under a run directory or a snapshot
store, starts no process and asks no backend. Five commands:

    runs       ROOT      per method: tasks complete/errored/in progress, attempts,
                         the attempt-1 rate, live attempts' turn counts, metrics
    task       ROOT TASK one task's fork tree as text, mermaid or json: baseline
                         attempts, analyst report, reviewer attempts, branches, trials
    export     JOURNAL   one trajectory as markdown or json, with a branch stitched
                         onto its parent up to the branch step
    snapshots  TARGET    every snapshot the journals name, which a plan selected,
                         and whether the podman store still has it
    list       DIR       every run root under a directory, with its state

Every command is safe beside a live run. A torn last line is skipped. A recorded
absolute path whose root has moved is found beside the file that names it. A
missing or changed parent is reported, not refused.
"""
