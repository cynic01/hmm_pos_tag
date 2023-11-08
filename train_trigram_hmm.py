#!/usr/bin/python
import re
import sys
from collections import defaultdict

"""
Implement a trigrm HMM here. 
You model should output the HMM similar to `train_hmm.py`.

Usage:  python train_trigram_hmm.py tags text > hmm-file

"""
OOV_WORD = "OOV"
INIT_STATE = "init"
FINAL_STATE = "final"

def train_trigram_hmm(tagFile, tokenFile, lam=0.01):
    vocab = set()
    emissions = defaultdict(lambda: defaultdict(int))
    unitrans = defaultdict(int)
    bitrans = defaultdict(lambda: defaultdict(int))
    tritrans = defaultdict(lambda: defaultdict(int))
    emissionsTotal = defaultdict(int)

    for tagString, tokenString in zip(tagFile, tokenFile):

        tags = re.split(r"\s+", tagString.rstrip())
        tokens = re.split(r"\s+", tokenString.rstrip())
        pairs = list(zip(tags, tokens))

        prev2, prev1 = INIT_STATE, INIT_STATE

        # update start probability for each sentence
        unitrans[INIT_STATE] += 1
        bitrans[INIT_STATE][INIT_STATE] += 1

        for (tag, token) in pairs:

            # this block is a little trick to help with out-of-vocabulary (OOV)
            # words.  the first time we see *any* word token, we pretend it
            # is an OOV.  this lets our model decide the rate at which new
            # words of each POS-type should be expected (e.g., high for nouns,
            # low for determiners).

            if token not in vocab:
                vocab.add(token)
                token = subcategorize(token)

            # increment the emission/transition observation
            emissions[tag][token] += 1
            emissionsTotal[tag] += 1

            unitrans[tag] += 1
            bitrans[prev1][tag] += 1
            tritrans[(prev2, prev1)][tag] += 1

            prev2, prev1 = prev1, tag

        # don't forget the stop probability for each sentence
        unitrans[FINAL_STATE] += 1
        bitrans[prev1][FINAL_STATE] += 1
        tritrans[(prev2, prev1)][FINAL_STATE] += 1

    tagSet = set(emissions.keys())
    initTagSet = tagSet | set((INIT_STATE,))
    finalTagSet = tagSet | set((FINAL_STATE,))

    # for (prev2, prev1) in tritrans:
    #     for tag in tritrans[(prev2, prev1)]:
    #         bigram_prob = bitrans[prev1][tag] / unitrans[prev1]
    #         trigram_prob = tritrans[(prev2, prev1)][tag] / bitrans[prev2][prev1]
    #         trans_prob = lam * trigram_prob + (1 - lam) * bigram_prob
    #         print(f"trans {prev2} {prev1} {tag} {trans_prob}")
    
    for prev2 in initTagSet:
        for prev1 in initTagSet:
            for tag in finalTagSet:
                bigram_prob = (bitrans[prev1][tag] + 1) / (unitrans[prev1] + len(finalTagSet))
                trigram_prob = (tritrans[(prev2, prev1)][tag] + 1) / (bitrans[prev2][prev1] + len(finalTagSet))
                trans_prob = lam * trigram_prob + (1 - lam) * bigram_prob
                print(f"trans {prev2} {prev1} {tag} {trans_prob}")

    for tag in emissions:
        for token in emissions[tag]:
            print(f"emit {tag} {token} {emissions[tag][token] / emissionsTotal[tag]} ")


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
        return OOV_WORD


if __name__ == "__main__":
    TAG_FILE = sys.argv[1]
    TOKEN_FILE = sys.argv[2]

    with open(TAG_FILE) as tag_file, open(TOKEN_FILE) as token_file:
        tags = tag_file.readlines()
        tokens = token_file.readlines()

        if len(tags) != len(tokens):
            raise ValueError("Length is different for two files!")
    
    train_trigram_hmm(tags, tokens)
