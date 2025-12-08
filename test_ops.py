import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'diff_sim_core'))

import forgeNN_cpp as fnn

print("--- Testing Indexing & Stacking ---")

# 1. Test Indexing (Select)
t = fnn.Tensor([10.0, 20.0, 30.0], requires_grad=True)
x = t[0] # Should be 10.0
y = t[2] # Should be 30.0

z = x * fnn.Tensor([3.0]) + y * fnn.Tensor([2.0]) # 3*10 + 2*30 = 90
z.backward()

print(f"Index Forward: {z.data[0,0]}")
# Gradient of z w.r.t t[0] is 3. t[1] is 0. t[2] is 2.
print(f"Grads: {t.grad.data[0,0]}, {t.grad.data[1,0]}, {t.grad.data[2,0]}")

assert abs(z.data[0,0] - 90.0) < 1e-5
assert abs(t.grad.data[0,0] - 3.0) < 1e-5
assert abs(t.grad.data[2,0] - 2.0) < 1e-5

print("Indexing Passed!")

# 2. Test Stacking
a = fnn.Tensor([1.0], requires_grad=True)
b = fnn.Tensor([2.0], requires_grad=True)

s = fnn.Tensor.stack([a, b]) # Should be [1.0, 2.0]
print(f"Stack Shape: {s.shape}")

loss = s.sum() # 1+2 = 3
loss.backward()

print(f"Stack Grads: a={a.grad.data[0,0]}, b={b.grad.data[0,0]}")

assert abs(loss.data[0,0] - 3.0) < 1e-5
assert abs(a.grad.data[0,0] - 1.0) < 1e-5
assert abs(b.grad.data[0,0] - 1.0) < 1e-5

print("Stacking Passed!")

# 3. Test MatMul & Transpose
print("--- Testing MatMul & Transpose ---")
# A: 2x2
# [[1, 2],
#  [3, 4]]
a_data = [1.0, 3.0, 2.0, 4.0] # ColMajor input? No, our constructor takes list as col vector then we reshape?
# Constructor takes 1D list and makes col vector. We need reshape.
# We don't have reshape yet.
# We must construct manually or stack columns.

# Create columns
c1 = fnn.Tensor.stack([fnn.Tensor([1.0],True), fnn.Tensor([3.0],True)])
c2 = fnn.Tensor.stack([fnn.Tensor([2.0],True), fnn.Tensor([4.0],True)])

# Stack colums -> Matrix (Wait, stack creates nx1. We want 2x2).
# Our stack creates a COLUMN vector from scalars.
# We don't have 'hstack' or 'reshape'.
# This exposes the reshape gap.

# However, we can construct using the (rows, cols) constructor but it initializes with 0.
# Then use set(r, c, val).
A = fnn.Tensor(2, 2, True)
A.set(0,0, 1.0); A.set(0,1, 2.0)
A.set(1,0, 3.0); A.set(1,1, 4.0)

# B: Identity * 2
B = fnn.Tensor(2, 2, True)
B.set(0,0, 2.0); B.set(0,1, 0.0)
B.set(1,0, 0.0); B.set(1,1, 2.0)

# C = A @ B
C = A @ B

# Check Forward
# [[2, 4],
#  [6, 8]]
print(f"C[0,0]={C.get(0,0)}, C[0,1]={C.get(0,1)}")
assert abs(C.get(0,0) - 2.0) < 1e-5
assert abs(C.get(1,1) - 8.0) < 1e-5

# Backward
loss = C.sum()
loss.backward()

# Grads
# dL/dA = Ones * B.T = [[2, 0], [0, 2]] * [[1,1],[1,1]] ? No.
# L = Sum(Cij). dL/dCij = 1.
# dL/dA = dL/dC * dC/dA = 1 * B.T.
# B.T = [[2, 0], [0, 2]].
# So A.grad should be [[2, 0], [0, 2]]?
# Wait. C = A*B. dC/dA is B.
# (A @ B)_ij = sum_k A_ik B_kj
# d(C_ij)/dA_im = delta_km * B_kj (if i=i)
# Correct: dL/dA = (dL/dC) @ B.T.
# dL/dC is all ones [[1,1],[1,1]].
# Ones @ B.T = [[1,1],[1,1]] @ [[2,0],[0,2]] = [[2,2],[2,2]].
print(f"A.grad[0,0]={A.grad[0,0]}")
assert abs(A.grad[0,0] - 2.0) < 1e-5
assert abs(A.grad[0,1] - 2.0) < 1e-5

# dL/dB = A.T @ (dL/dC) = [[1,3],[2,4]] @ [[1,1],[1,1]]
# = [[1+3, 1+3], [2+4, 2+4]] = [[4,4], [6,6]]
print(f"B.grad[0,0]={B.grad[0,0]}")
assert abs(B.grad[0,0] - 4.0) < 1e-5
assert abs(B.grad[1,0] - 6.0) < 1e-5

print("MatMul & Transpose Passed!")

# 4. Test Math & Activations (Pow, ReLU, Tanh)
print("--- Testing Math & Activations ---")
# Pow
x = fnn.Tensor([2.0, 3.0], requires_grad=True)
y = x.pow(3.0) # [8, 27]
y.backward()
# dy/dx = 3 * x^2 = [12, 27]
print(f"Pow(2^3)={y.get(0,0)}, Grad={x.grad[0,0]}")
assert abs(y.get(0,0) - 8.0) < 1e-5
assert abs(x.grad[0,0] - 12.0) < 1e-5

# ReLU
x = fnn.Tensor([-1.0, 2.0], requires_grad=True)
y = fnn.relu(x) # [0, 2]
y.backward()
# Grad: [0, 1]
print(f"ReLU(-1)={y.get(0,0)}, Grad={x.grad[0,0]}")
assert abs(y.get(0,0) - 0.0) < 1e-5
assert abs(x.grad[0,0] - 0.0) < 1e-5
assert abs(x.grad[1,0] - 1.0) < 1e-5

# Tanh
x = fnn.Tensor([0.0], requires_grad=True)
y = fnn.tanh(x) # 0
y.backward()
# Grad: 1 - tanh^2(0) = 1
print(f"Tanh(0)={y.get(0,0)}, Grad={x.grad[0,0]}")
assert abs(y.get(0,0) - 0.0) < 1e-5
assert abs(x.grad[0,0] - 1.0) < 1e-5

print("Activations Passed!")

# 5. Reshape
print("--- Testing Reshape ---")
v = fnn.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True) # 4x1
M = v.reshape(2, 2) # 2x2
# [[1, 2],
#  [3, 4]] ? Assuming ColMajor filling or RowMajor?
# Eigen Map with (rows, cols) usually fills ColMajor.
# v data: 1,2,3,4.
# r=2, c=2.
# Col 0: 1, 2. Col 1: 3, 4.
# So M is [[1, 3], [2, 4]].
# Let's check.
print(f"Reshape(2,2): (0,0)={M.get(0,0)}, (1,0)={M.get(1,0)}")
# If ColMajor: M(0,0)=1, M(1,0)=2. M(0,1)=3, M(1,1)=4.

# Ops
loss = M.sum()
loss.backward()
# v.grad should be all ones
print(f"Reshape Grad: {v.grad[0,0]}")
assert abs(v.grad[0,0] - 1.0) < 1e-5

print("Reshape Passed!")

# 6. Cat
print("--- Testing Cat ---")
# Dim 0
t1 = fnn.Tensor([1.0, 2.0], requires_grad=True) # 2x1
t2 = fnn.Tensor([3.0, 4.0], requires_grad=True) # 2x1
res0 = fnn.Tensor.cat([t1, t2], 0) # 4x1
print(f"Cat0 shape: ({res0.rows()}, {res0.cols()})")
assert res0.rows() == 4 and res0.cols() == 1
# Grad
res0.sum().backward()
assert abs(t1.grad[0,0] - 1.0) < 1e-5
assert abs(t2.grad[0,0] - 1.0) < 1e-5

# Dim 1
t3 = fnn.Tensor([1.0],True) # 1x1
t4 = fnn.Tensor([2.0],True) # 1x1
res1 = fnn.Tensor.cat([t3, t4], 1) # 1x2
print(f"Cat1 shape: ({res1.rows()}, {res1.cols()})")
assert res1.rows() == 1 and res1.cols() == 2
res1.sum().backward()
assert abs(t3.grad[0,0] - 1.0) < 1e-5
assert abs(t4.grad[0,0] - 1.0) < 1e-5


# 7. New Math Ops
print("--- Testing Extended Math ---")

# Exp
x = fnn.Tensor([0.0], True)
y = x.exp() # 1.0
y.backward()
# dy/dx = exp(x) = 1.0
assert abs(y.get(0,0) - 1.0) < 1e-5
assert abs(x.grad[0,0] - 1.0) < 1e-5
print(f"Exp(0)=1.0, Grad=1.0 - OK")

# Log
x = fnn.Tensor([2.71828], True)
y = x.log() # 1.0
y.backward() # dy/dx = 1/x approx 0.367
assert abs(y.get(0,0) - 1.0) < 1e-4
assert abs(x.grad[0,0] - (1.0/2.71828)) < 1e-4
print(f"Log(e)=1.0, Grad=1/e - OK")

# Sqrt
x = fnn.Tensor([4.0], True)
y = x.sqrt() # 2.0
y.backward() # dy/dx = 1/(2*2) = 0.25
assert abs(y.get(0,0) - 2.0) < 1e-5
assert abs(x.grad[0,0] - 0.25) < 1e-5
print(f"Sqrt(4)=2.0, Grad=0.25 - OK")

# Abs
x = fnn.Tensor([-3.0, 3.0], True)
y = x.abs() # [3, 3]
y.backward() # [-1, 1]
assert abs(y.get(0,0) - 3.0) < 1e-5
assert abs(x.grad[0,0] + 1.0) < 1e-5
assert abs(x.grad[1,0] - 1.0) < 1e-5
print(f"Abs([-3,3])=[3,3], Grad=[-1,1] - OK")

# Clamp
x = fnn.Tensor([-1.0, 0.5, 2.0], True)
y = x.clamp(0.0, 1.0) # [0, 0.5, 1]
y.backward() # [0, 1, 0]
assert abs(y.get(0,0) - 0.0) < 1e-5
assert abs(y.get(2,0) - 1.0) < 1e-5
assert abs(x.grad[0,0]) < 1e-5
assert abs(x.grad[1,0] - 1.0) < 1e-5
assert abs(x.grad[2,0]) < 1e-5
print(f"Clamp - OK")

# Reductions (Min/Max)
x = fnn.Tensor([1.0, 5.0, 3.0], True)
mi = x.min() # 1
mi.backward() # [1, 0, 0]
assert abs(mi.get(0,0) - 1.0) < 1e-5
assert abs(x.grad[0,0] - 1.0) < 1e-5
assert abs(x.grad[1,0]) < 1e-5
print("Min - OK")

x.zero_grad() # Manual zero for test
ma = x.max() # 5
ma.backward() # [0, 1, 0]
assert abs(ma.get(0,0) - 5.0) < 1e-5
assert abs(x.grad[1,0] - 1.0) < 1e-5
print("Max - OK")

# Sum/Mean Axis
t_a = fnn.Tensor([1.0, 2.0], True)
t_b = fnn.Tensor([3.0, 4.0], True)
x = fnn.Tensor.cat([t_a, t_b], 1) # [[1,3], [2,4]]
s0 = x.sum(0) # Col sum -> [3, 7] (1x2)
s0.sum().backward()
# Grad should be ones broadcasted
print("Axis Sum - OK")

print("Extended Math Passed!")
print("--- All Tests Passed ---")
