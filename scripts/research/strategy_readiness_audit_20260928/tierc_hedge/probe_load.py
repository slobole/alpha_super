import time, traceback
import tc_common as c
import pandas as pd
for name, fn in (("vixm", c.load_vixm), ("trin", c.load_trin), ("ctc", c.load_ctc)):
    t=time.time()
    try:
        df = fn()
        print(name, df.shape, df.index[0], df.index[-1], round(time.time()-t,1))
        syms = sorted(set(df.columns.get_level_values(0)))
        print(syms)
        print(sorted(set(df.columns.get_level_values(1))))
    except Exception as e:
        traceback.print_exc()
