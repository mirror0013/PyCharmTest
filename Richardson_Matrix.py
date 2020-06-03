import numpy as np

np.random.seed()

n = 100
x0, x1 = -2, 2
A = np.random.rand(n, n) * (x1 - x0) + x0
D = A.T.dot(A)
n4 = n // 4
for i in range(n4, n):
    D[i - n4, i:n] = 0
    D[i:n, i - n4] = 0
print(f"Matrix A shape = {D.shape}")

x0, x1, n2 = 1, 10, n // 2
x = np.zeros(n)
x[:n2] = np.random.rand(n2) * (x1 - x0) + x0
x[n2:] = np.flip(x[:n2])
print(f"Vector x shape = {x.shape}")

b = D.dot(x)
print(f"Vector b shape = {b.shape}")

np.savetxt("Richardson-Matrix-A.txt", D, fmt=' %8.4f', header=str(n))
np.savetxt("Richardson-Vector-b.txt", b, fmt=' %8.4f', \
           delimiter=" ", newline=' ', header=str(n))
np.savetxt("Richardson-Vector-x.txt", x, fmt=' %8.4f', \
           delimiter=" ", newline=' ', header=str(n))
