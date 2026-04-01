from heapq import nlargest
from joblib import Parallel, delayed
Parallel(n_jobs=2)(
delayed(nlargest)(2, n) for n in (range(4), 'abcde'))