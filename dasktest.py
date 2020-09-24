import dask.array as da
import numpy as np
import time

start = time.time()

x = da.random.normal(10, 0.1, size=(20000, 20000), 
  chunks=(1000,1000)
  )
y = x.mean(axis=0)
print(y.compute())
print(time.time()-start, 'seconds')

start = time.time()

x = np.random.normal(10, 0.1, size=(20000, 20000))
y = np.mean(x)
print(y)
print(time.time()-start, 'seconds')