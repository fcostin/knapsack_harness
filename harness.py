#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test harness to evaluate given knapsack solver over given test cases

usage example:

    python harness.py my_awesome_solver.py data/ --time-limit 60 --mem-limit 2000000000
"""

import sys
import os
import argparse
from collections import namedtuple
import numpy
import subprocess
import resource
from multiprocessing.pool import ThreadPool
from itertools import product, izip
from jinja2 import Template

Problem = namedtuple('Problem', ['k', 'v', 'w', 'file_name'])

def load_problem(file_name):
    with open(file_name, 'r') as f:
        data = ''.join(list(f))
    lines = data.split('\n')
    firstline = lines[0].split()
    n = int(firstline[0])
    k = int(firstline[1])
    v = []
    w = []
    for i in range(1, n+1):
        line = lines[i]
        parts = line.split()
        v.append(int(parts[0]))
        w.append(int(parts[1]))
    assert len(v) == len(w) == n
    return Problem(k, numpy.asarray(v), numpy.asarray(w), file_name)


problem_order = lambda p : (len(p.v), p.file_name)

fmt_problem = lambda p : 'Problem{n=%d, capacity=%d, "%s"}' % (len(p.v), p.k, p.file_name)


def check_feasibility(problem, soln):
    n = len(problem.v)
    k = 0
    if len(soln) != n:
        return False
    for j, took in enumerate(soln):
        if took:
            k += problem.w[j]
            if k > problem.k:
                return False
    return True

def measure_obj(problem, soln):
    return sum(problem.v[j] for j in soln)

Result = namedtuple('Result', ['status', 'obj', 'soln'])


def limited_exec(solver_file_name, problem_file_name, time_limit, mem_limit):

    def limit_solve_resources():
        # set (soft, hard) resource limits
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, )*2) # memory address space (bytes)
        resource.setrlimit(resource.RLIMIT_CPU, (time_limit, )*2) # cpu time limit (seconds)
    
    p = subprocess.Popen(['python', solver_file_name, problem_file_name], shell=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=limit_solve_resources)
    out, err = p.communicate()
    return p.returncode, out, err

def maybe_parse_soln(output):
    lines = [x.strip() for x in output.split('\n') if x.strip()]
    if len(lines) < 2:
        return None
    else:
        obj_toks = lines[-2].strip().split()
        soln_toks = lines[-1].strip().split()
        if not obj_toks or not soln_toks:
            return None
        try:
            obj = int(obj_toks[0])
            soln = map(int, soln_toks)
        except ValueError:
            return None
        return (obj, soln)


def solve(solver_file_name, problem, time_limit, mem_limit):
    retval, out, err = limited_exec(solver_file_name, problem.file_name, time_limit, mem_limit)
    if retval:
        r = Result('ABORTED', obj=None, soln=None)
    else:
        maybe_soln = maybe_parse_soln(out)
        if maybe_soln is None:
            r = Result('PARSE_ERROR', obj=None, soln=None)
        else:
            obj, soln = maybe_soln
            feasible = check_feasibility(problem, soln)
            if not feasible:
                r = Result('INFEASIBLE', obj=None, soln=soln)
            else:
                true_obj = measure_obj(problem, soln)
                r = Result('FEASIBLE', obj=true_obj, soln=soln)
    print 'job completed: {"%s", "%s", %s}' % (solver_file_name, problem.file_name, r.status)
    return r

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('problem_dir', metavar='PROBLEM_DIR', nargs=1)
    p.add_argument('solvers', metavar='SOLVER_PY', nargs='+')
    p.add_argument('--time-limit', metavar='t', type=int, default=60, help='max cpu time (seconds)')
    p.add_argument('--mem-limit', metavar='m', type=int, default=2000000000, help='max mem (bytes)')
    p.add_argument('-n', metavar='NPROCS', type=int, default=4, help='number of solves to test in parallel')
    p.add_argument('--report-template', metavar='FILE', default='report.template')
    p.add_argument('--out', metavar='FILE', default='report.html')
    return p.parse_args(argv)


def collect(file_names):
    bucket = []
    for file_name in file_names:
        if os.path.isfile(file_name):
            bucket.add(file_name)
        elif os.path.isdir(file_name):
            bucket += [os.path.join(file_name, x) for x in os.listdir(file_name)]
    return bucket

Job = namedtuple('Job', ['solver', 'problem', 'time_limit', 'mem_limit'])
job_key = lambda job : (job.solver, job.problem.file_name)

fmt_result = lambda result : result.obj if result.status == 'FEASIBLE' else result.status

def strip_common_prefix(names):
    if not names:
        return names
    if not names[0]:
        return names
    c = names[0][0]
    can_strip = all(name.startswith(c) for name in names)
    if can_strip:
        return strip_common_prefix([name[1:] for name in names])
    else:
        return names

def make_nicknames(names):
    return dict(izip(names, strip_common_prefix(names)))

def main():
    args = parse_args(sys.argv[1:])
    problems = sorted(map(load_problem, collect(args.problem_dir)), key=problem_order)

    print 'collected %d problems from "%s"' % (len(problems), args.problem_dir[0])

    jobs = [Job(s, p, args.time_limit, args.mem_limit) for p,s in product(problems, args.solvers)]

    print 'got %d test jobs' % len(jobs)

    print 'running test jobs:'
    pool = ThreadPool(args.n)
    results = pool.map(lambda args : solve(*args), jobs)
    job_results = dict(izip(map(job_key, jobs), results))

    fmt_s = make_nicknames(args.solvers)
    fmt_p = make_nicknames([p.file_name for p in problems])

    rows = [['Problem'] + [fmt_s[s] for s in args.solvers]]
    for p in problems:
        solver_results = [fmt_result(job_results[(s, p.file_name)]) for s in args.solvers]
        rows.append([fmt_p[p.file_name]] + solver_results)

    # render it
    with open(args.report_template, 'r') as f_template:
        t = Template(f_template.read())
    with open(args.out, 'w') as f_out:
        f_out.write(t.render(rows=rows))



if __name__ == '__main__':
    main()

