knapsack solver test harness
============================

what
----

benchmark a list of solvers over a directory of test problems, with resource limits for CPU and memory per solve, in parallel.

usage
-----

    harness.py [-h] [--time-limit t] [--mem-limit m] [-n NPROCS]
            [--report-template FILE] [--out FILE]
            PROBLEM_DIR SOLVER_PY [SOLVER_PY ...]

example
-------

run your solvers for at most 60s using at most 1GB (per solve), with 4 solves at once

    python harness.py problem_data/ my_awesome_solver.py my_laughable_solver.py --mem-limit 1000000000 --time-limit 60 -n 4

progress is spammed to stdout as jobs complete:

    collected 18 problems from "data/"
    got 36 test jobs
    running test jobs:
    job completed: {"../knapsack/solver_sparse.py", "data/ks_4_0", FEASIBLE}
    job completed: {"../knapsack/solver_dense.py", "data/ks_4_0", FEASIBLE}
    job completed: {"../knapsack/solver_sparse.py", "data/ks_19_0", FEASIBLE}
    job completed: {"../knapsack/solver_dense.py", "data/ks_19_0", FEASIBLE}
    job completed: {"../knapsack/solver_sparse.py", "data/ks_30_0", FEASIBLE}
    ...

an ugly html summary is written to `report.html`:

![ugly_example_results](/example/my_eyes.png "Table of solver results")


todo
----

1.  make report less eye-damaging
2.  distinguish between memory limit, time limit, and other failure modes
3.  display cpu time and memory usage in report?

