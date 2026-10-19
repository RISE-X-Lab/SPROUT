"""The config files of a host: written from the packaged templates, and checked.

``init`` writes each missing file from its template, with the values of this host. ``doctor`` lists
each problem in runstore.json that makes jobs fail, with its fix. ``setup`` guides through the host
checks, the config files, the model endpoint and the start of the services, and asks before each
change. docs/SERVICES.md ("Guided setup") describes the steps.
"""
