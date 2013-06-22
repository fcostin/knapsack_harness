knapsack solver test harness
============================

usage
-----

    harness.py [-h] [--time-limit t] [--mem-limit m] [-n NPROCS]
               SOLVER_PY PROBLEM_FILE [PROBLEM_FILE ...]

example
-------

run your awesome solver for at most 60s using at most 2GB (per solve), with 3 solves at once

    python harness.py my_awesome_solver.py data/ --time-limit 60 --mem-limit 2000000000 -n 3

todo
----

1.  write a nice test report at the end instead of spamming to stdout
2.  distinguish between memory limit, time limit, and other failure modes

