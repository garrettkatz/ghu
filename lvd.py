import numpy as np

def lvd(s, t):

    m, n = len(s), len(t)
    d = np.empty((m+1,n+1), dtype=int)
    d[:,0] = np.arange(m+1)
    d[0,:] = np.arange(n+1)
    
    for j in range(n):
        for i in range(m):
            if s[i] == t[j]: d[i+1,j+1] = d[i,j]
            else: d[i+1,j+1] = min(
                d[i, j+1] + 1,
                d[i+1, j] + 1,
                d[i, j] + 1)

    return d[m,n]

if __name__ == "__main__":

    assert(lvd("", "kitten") == 6)
    assert(lvd("kitten","kitten") == 0)
    assert(lvd("kitten","mitten") == 1)
    assert(lvd("kitten","mittens") == 2)
    assert(lvd("kitten","mitte") == 2)
    assert(lvd(list("kitten"),"mitten") == 1)
    assert(lvd(range(4), range(1,5)) == 2)
    assert(lvd(["cat","hat"], ["cat"]) == 1)

