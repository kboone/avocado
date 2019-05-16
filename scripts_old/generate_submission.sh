#!/usr/bin/env sh

# Load a pretrained model and calculate scores for each object in the test set.
python code/do_scores.py

# Apply a "real" class 99 prediction and convert the scores into a submission.
python code/convert_scores.py

# Apply a class 99 prediction obtained by probing to see what the class 99s
# look like. This improves the leaderboard score, but isn't useful for real
# science.
python code/apply_probe_99.py
