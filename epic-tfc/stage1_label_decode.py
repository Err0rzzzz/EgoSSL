import os
import pandas as pd

ANN_ROOT  = r"F:/EgoSSL/epic-tfc/annotations/EPIC-Kitchens-100-Annotations"
verb_csv = os.path.join(ANN_ROOT, "EPIC_100_verb_classes.csv")
noun_csv = os.path.join(ANN_ROOT, "EPIC_100_noun_classes.csv")

verb_id = 1
noun_id = 19

verbs = pd.read_csv(verb_csv)
nouns = pd.read_csv(noun_csv)

# column names vary slightly; try common options
vcol = "verb" if "verb" in verbs.columns else verbs.columns[-1]
ncol = "noun" if "noun" in nouns.columns else nouns.columns[-1]

verb_word = verbs.loc[verbs["id"] == verb_id, vcol].values[0]
noun_word = nouns.loc[nouns["id"] == noun_id, ncol].values[0]

print("verb:", verb_id, "->", verb_word)
print("noun:", noun_id, "->", noun_word)
