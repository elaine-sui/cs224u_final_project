import pickle
import pandas as pd

# TODO: aggregate consistency results:
"""
Forward + backward
Forward negated + unnegated
Backward negated + unnegated
Forward randomized (1234, 5678, 910)
Backward randomized (1234, 5678, 910)

How to aggregate CoT?
 - Intersection of both CoTs (converted so that they are comparable)
 - Union of both CoTs (converted so that they are comparable)
 - Choose the longer one of the two

How to aggregate answer?
 - If agree, take that answer
 - If disagreement, choose answer corresponding with the longest CoT
"""