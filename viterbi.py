#!/usr/bin/python
import re
import sys
import math

"""
Implement the Viterbi algorithm in Python (no tricks other than logmath!), given an
HMM, on sentences, and outputs the best state path.
Please check `viterbi.pl` for reference.

Usage:  python viterbi.py hmm-file < text > tags

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
States = {}
Voc = {}

try:
    with open(hmmfile, 'r') as HMM:
        for line in HMM:
            trans_match = re.match(r"trans\s+(\S+)\s+(\S+)\s+(\S+)", line)
            if trans_match:
                qq, q, p = trans_match.groups()
                p_trans.setdefault(qq, {})[q] = math.log(float(p))
                States[qq] = 1
                States[q] = 1
            else:
                emit_match = re.match(r"emit\s+(\S+)\s+(\S+)\s+(\S+)", line)
                if emit_match:
                    q, w, p = emit_match.groups()
                    p_emit.setdefault(q, {})[w] = math.log(float(p))
                    States[q] = 1
                    Voc[w] = 1
except IOError:
    print(f"could not open {hmmfile}")
    exit(1)

for line in sys.stdin:
    line = line.strip()
    w = line.split()
    w.insert(0, "")
    n = len(w) - 1
    p_viterbi = {}
    Backtrace = {}
    p_viterbi[0] = {init_state: 0.0}
    for i in range(1, n + 1):
        if w[i] not in Voc:
            if verbose:
                sys.stderr.write(f"OOV: {w[i]}\n")
            w[i] = OOV_symbol
        for q in States.keys():
            for qq in States.keys():
                if qq in p_trans and q in p_trans[qq] and q in p_emit and w[i] in p_emit[q] and i-1 in p_viterbi and qq in p_viterbi[i-1]:
                    v = p_viterbi[i-1][qq] + p_trans[qq][q] + p_emit[q][w[i]]
                    if i not in p_viterbi or q not in p_viterbi[i] or v > p_viterbi[i][q]:
                        if i not in p_viterbi: p_viterbi[i] = {}
                        p_viterbi[i][q] = v
                        if i not in Backtrace: Backtrace[i] = {}
                        Backtrace[i][q] = qq
            if verbose:
                sys.stderr.write(f"V[{i}, {q}] = {p_viterbi[i][q]} ({p_emit[i][q]})\n")

    foundgoal = False
    for qq in States.keys():
        if qq in p_trans and final_state in p_trans[qq] and n in p_viterbi and qq in p_viterbi[n]:
            v = p_viterbi[n][qq] + p_trans[qq][final_state]
            if not foundgoal or v > goal:
                goal = v
                foundgoal = True
                q = qq

    if foundgoal:
        t = []
        for i in range(n, 0, -1):
            t.insert(0, q)
            q = Backtrace[i][q]
    if verbose:
        sys.stderr.write(f"{math.exp(goal)}\n")
    if foundgoal:
        print(" ".join(t), end='')
    print('\n', end='')