import numpy as np
import pandas as pd

rowheaders = [
    ('A', 't1'),
    ('A', 't2'),
    ('A', 't2'),
    ('B', 't2'),
    ('B', 't3'),
    ('C', 't1'),
    ('Z', 't2'),
    ('Z', 't4'),
    ('D', 't2'),
    ('D', 't3'),
]

pd.MultiIndex.from_tuples( rowheaders )

df = pd.DataFrame( np.random.rand(len(rowheaders), 3), columns=['a', 'b', 'c'], index=pd.MultiIndex.from_tuples( rowheaders ))

df.index.set_names(['cust', 'tier'], inplace=True)

idxdf = pd.DataFrame( df.index.values.tolist(), columns=df.index.names  )

inc = (idxdf != idxdf.shift()).cumsum(axis=1) > 0

# x = inc['cust']

def find_span( x ):
    idx = x.index[x].to_series()
    x = x * (idx.shift(-1).fillna(len(x)) - idx)
    return x.fillna(0.).astype(np.int64)

param = inc.apply(find_span)

class H(object):
    def __init__(self, rowspan=1):
        self.rowspan = rowspan
    def __repr__(self):
        return '<H>: rowspan=%s'%self.rowspan

map( lambda r: map( lambda x: H(x) if x > 0 else None, r), param.values )
