#!/usr/bin/python
import re
import sys
import math

"""
Implement the trigram Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.

Usage:  python trigram_viterbi.py hmm-file < text > tags

special keywords:
 $init_state   (an HMM state) is the single, silent start state
 $final_state  (an HMM state) is the single, silent stop state
 $OOV_symbol   (an HMM symbol) is the out-of-vocabulary word
"""

init_state = 'init'
final_state = 'final'
OOV_symbol = 'OOV'

verbose = False

# read in the HMM and store the probabilities as log probabilities

hmmfile = sys.argv[1]

p_trans = {}
p_emit = {}
States = set()
Voc = set()

try:
    with open(hmmfile, 'r') as HMM:
        for line in HMM:
            trans_match = re.match(r"trans\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
            if trans_match:
                q2, q1, q, p = trans_match.groups()
                p_trans.setdefault((q2, q1), {})[q] = math.log(float(p))
                States.update((q2, q1, q))
            else:
                emit_match = re.match(r"emit\s+(\S+)\s+(\S+)\s+(\S+)", line)
                if emit_match:
                    q, w, p = emit_match.groups()
                    p_emit.setdefault(q, {})[w] = math.log(float(p))
                    States.add(q)
                    Voc.add(w)
except IOError:
    print(f"could not open {hmmfile}")
    exit(1)

for line in sys.stdin:
    line = line.strip()
    w = line.split()
    w.insert(0, "")
    w.insert(0, "")
    n = len(w) - 1
    p_viterbi = {}
    Backtrace = {}
    p_viterbi[0] = {init_state: 0.0}
    p_viterbi[1] = {init_state: 0.0}
    for i in range(2, n + 1):
        if w[i] not in Voc:
            if verbose:
                sys.stderr.write(f"OOV: {w[i]}\n")
            w[i] = OOV_symbol
        for q in States:
            for q1 in States:
                for q2 in States:
                    if (q2, q1) in p_trans and q in p_trans[(q2, q1)] \
                        and q in p_emit and w[i] in p_emit[q] \
                        and i-1 in p_viterbi and q1 in p_viterbi[i-1]:
                        v = p_viterbi[i-1][q1] + p_trans[(q2, q1)][q] + p_emit[q][w[i]]
                        if i not in p_viterbi or q not in p_viterbi[i] or v > p_viterbi[i][q]:
                            if i not in p_viterbi: p_viterbi[i] = {}
                            p_viterbi[i][q] = v
                            if i not in Backtrace: Backtrace[i] = {}
                            Backtrace[i][q] = q1
            if verbose:
                sys.stderr.write(f"V[{i}, {q}] = {p_viterbi[i][q]} ({p_emit[i][q]})\n")

    foundgoal = False
    for q1 in States:
        for q2 in States:
            if (q2, q1) in p_trans and final_state in p_trans[(q2, q1)] and n in p_viterbi and q1 in p_viterbi[n]:
                v = p_viterbi[n][q1] + p_trans[(q2, q1)][final_state]
                if not foundgoal or v > goal:
                    goal = v
                    foundgoal = True
                    q = q1

    if foundgoal:
        t = []
        for i in range(n, 1, -1):
            t.insert(0, q)
            q = Backtrace[i][q]
    if verbose:
        sys.stderr.write(f"{math.exp(goal)}\n")
    if foundgoal:
        print(" ".join(t), end='')
    print('\n', end='')