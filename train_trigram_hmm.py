#!/usr/bin/python
import re
import sys
from collections import defaultdict

"""
Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm.py tags text > hmm-file

"""
def train_trigram_hmm(tagFile, tokenFile):
    vocab = set()
    OOV_WORD = "OOV"
    INIT_STATE = "init"
    FINAL_STATE = "final"

    emissions = {}
    transitions = {}
    transitionsTotal = defaultdict(int)
    emissionsTotal = defaultdict(int)

    for tagString, tokenString in zip(tagFile, tokenFile):

        tags = re.split("\s+", tagString.rstrip())
        tokens = re.split("\s+", tokenString.rstrip())
        pairs = list(zip(tags, tokens))

        prev1, prev2 = INIT_STATE, INIT_STATE

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab.add(token)
                token = OOV_WORD

            if tag not in emissions:
                emissions[tag] = defaultdict(int)
            if (prev1, prev2) not in transitions:
                transitions[(prev1, prev2)] = defaultdict(int)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            transitions[(prev1, prev2)][tag] += 1
            transitionsTotal[(prev1, prev2)] += 1

            prev1, prev2 = prev2, tag

        # don't forget the stop probability for each sentence
        if (prev1, prev2) not in transitions:
            transitions[(prev1, prev2)] = defaultdict(int)

        transitions[(prev1, prev2)][FINAL_STATE] += 1
        transitionsTotal[(prev1, prev2)] += 1

    tagSet = set(emissions)
    initTagSet = tagSet | set((INIT_STATE,))
    finalTagSet = tagSet | set((FINAL_STATE,))
    
    # for (prev1, prev2) in transitions:
    #     for tag in transitions[(prev1, prev2)]:
    for prev1 in initTagSet:
        for prev2 in initTagSet:
            for tag in finalTagSet:
                print(f"trans {prev1} {prev2} {tag} {(transitions.get((prev1, prev2), {}).get(tag, 0) + 1) / (transitionsTotal[(prev1, prev2)] + len(tagSet))}")

    for tag in emissions:
        # for token in emissions[tag]:
        for token in vocab | set((OOV_WORD,)):
            print(f"emit {tag} {token} {(emissions[tag][token] + 1) / (emissionsTotal[tag] + len(vocab))}")


if __name__ == "__main__":
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]

    with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
        tags = tag_file.readlines()
        tokens = token_file.readlines()

        if len(tags) != len(tokens):
            raise ValueError("Length is different for two files!")
    
    train_trigram_hmm(tags, tokens)
