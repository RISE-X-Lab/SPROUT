"""Read-only inspection of what a run leaves on disk.

``python -m sprout.inspect`` reads journals, receipts, summaries and plans and
prints what they say; it writes nothing under a run directory or a snapshot
store, starts no process and asks no backend. Five commands:

    runs       ROOT      per method: tasks complete/errored/in progress, attempts,
                         the attempt-1 rate, live attempts' turn counts, metrics
    task       ROOT TASK the fork tree of one task: baseline attempts, analyst
                         report, reviewer attempts, branches and their trials
                         (text, mermaid or json)
    export     JOURNAL   one trajectory as markdown or json; a branch is stitched
                         onto its parent up to the branch step
    snapshots  TARGET    every snapshot the journals name, which a plan selected,
                         and whether the podman store still has it
    list       DIR       every run root under a directory, with its state

Everything is safe beside a live run: a torn last line is skipped, recorded
absolute paths are re-rooted beside the file that names them when a root has
moved, and a missing or changed parent is reported rather than refused.
"""
