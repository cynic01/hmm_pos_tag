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

def subcategorize(word):
    if not re.search(r'\w', word):
        return 'PUNCT'
    elif re.search(r'[A-Z]', word):
        return 'CAPITAL'
    elif re.search(r'\d', word):
        return 'NUM'
    elif re.search(r'(ion\b|ty\b|ics\b|ment\b|ence\b|ance\b|ness\b|ist\b|ism\b)', word):
        return 'NOUNLIKE'
    elif re.search(r'(ate\b|fy\b|ize\b|\ben|\bem)', word):
        return 'VERBLIKE'
    elif re.search(r'(\bun|\bin|ble\b|ry\b|ish\b|ious\b|ical\b|\bnon)', word):
        return 'JJLIKE'
    else:
        return OOV_symbol


for line in sys.stdin:
    w = line.strip().split()
    w = ["", ""] + w
    n = len(w) - 1
    p_viterbi = {}
    Backtrace = {}
    p_viterbi[0] = {init_state: 0.0}
    p_viterbi[1] = {init_state: 0.0}
    for i in range(2, n + 1):
        if w[i] not in Voc:
            if verbose:
                sys.stderr.write(f"OOV: {w[i]}\n")
            w[i] = subcategorize(w[i])
        for q in States:
            for (q2, q1), p_state in p_trans.items():
                if q in p_state \
                    and q in p_emit and w[i] in p_emit[q] \
                    and i-1 in p_viterbi and q1 in p_viterbi[i-1]:
                    v = p_viterbi[i-1][q1] + p_state[q] + p_emit[q][w[i]]
                    if i not in p_viterbi or q not in p_viterbi[i] or v > p_viterbi[i][q]:
                        p_viterbi.setdefault(i, {})[q] = v
                        Backtrace.setdefault(i, {})[q] = q1
            if verbose:
                sys.stderr.write(f"V[{i}, {q}] = {p_viterbi[i][q]} ({p_emit[q][w[i]]})\n")

    foundgoal = False
    for (q2, q1), p_state in p_trans.items():
        if final_state in p_state and n in p_viterbi and q1 in p_viterbi[n]:
            v = p_viterbi[n][q1] + p_state[final_state]
            if not foundgoal or v > goal:
                goal = v
                foundgoal = True
                q = q1

    if verbose:
        sys.stderr.write(f"{math.exp(goal)}\n")

    if foundgoal:
        t = []
        for i in range(n, 1, -1):
            t.insert(0, q)
            q = Backtrace[i][q]
        print(" ".join(t), end='')
    print('\n', end='')