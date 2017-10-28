import numpy as np

x = np.random.randint(2, size=(100,2))
y = np.array(list(map(lambda s: s[0]^s[1], x)))
y = y.reshape(100,1)

data = np.column_stack((x,y))

np.save("output.out", data)