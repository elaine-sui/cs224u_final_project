import pickle
import pandas as pd

# TODO: aggregate consistency results:
"""
Baseline (1234, 5678, 910)
Forward + backward
Forward negated + unnegated
Backward negated + unnegated
Forward randomized (1234, 5678, 910)
Backward randomized (1234, 5678, 910)

How to aggregate CoT? (write each one as a separate function -- helpful for evaluation/analysis)
 - Intersection of all CoTs (converted so that they are comparable)
 - Union of all CoTs (converted so that they are comparable)
 - Choose the longest one

How to aggregate answer? (write each one as a separate function -- helpful for evaluation/analysis)
 - Take majority answer
 - [Only applies if aggregating 2 results] 
    - If disagreement, 2 choices:
        - 1) Must choose answer: choose answer corresponding with the longest CoT
        - 2) Can output "I don't know"
"""