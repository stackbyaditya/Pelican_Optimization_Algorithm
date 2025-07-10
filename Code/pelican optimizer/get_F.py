import numpy as np

# Get_F function
def Get_F(F):
    if F == 'F1':
        F_obj = F1
        LB = -100
        UB = 100
        Dim = 10
    elif F == 'F2':
        F_obj = F2
        LB = -10
        UB = 10
        Dim = 10
    elif F == 'F3':
        F_obj = F3
        LB = -100
        UB = 100
        Dim = 10
    elif F == 'F4':
        F_obj = F4
        LB = -100
        UB = 100
        Dim = 10
    elif F == 'F5':
        F_obj = F5
        LB = -30
        UB = 30
        Dim = 10
    elif F == 'F6':
        F_obj = F6
        LB = -100
        UB = 100
        Dim = 10
    elif F == 'F7':
        F_obj = F7
        LB = -1.28
        UB = 1.28
        Dim = 10
    elif F == 'F8':
        F_obj = F8
        LB = -500
        UB = 500
        Dim = 10
    elif F == 'F9':
        F_obj = F9
        LB = -5.12
        UB = 5.12
        Dim = 10
    elif F == 'F10':
        F_obj = F10
        LB = -32
        UB = 32
        Dim = 10
    elif F == 'F11':
        F_obj = F11
        LB = -600
        UB = 600
        Dim = 10
    elif F == 'F12':
        F_obj = F12
        LB = -50
        UB = 50
        Dim = 10
    elif F == 'F13':
        F_obj = F13
        LB = -50
        UB = 50
        Dim = 10
    elif F == 'F14':
        F_obj = F14
        LB = -65.536
        UB = 65.536
        Dim = 2
    elif F == 'F15':
        F_obj = F15
        LB = -5
        UB = 5
        Dim = 4
    elif F == 'F16':
        F_obj = F16
        LB = -5
        UB = 5
        Dim = 2
    elif F == 'F17':
        F_obj = F17
        LB = np.array([-5, 0])
        UB = np.array([10, 15])
        Dim = 2
    elif F == 'F18':
        F_obj = F18
        LB = -2
        UB = 2
        Dim = 2
    elif F == 'F19':
        F_obj = F19
        LB = 0
        UB = 1
        Dim = 3
    elif F == 'F20':
        F_obj = F20
        LB = 0
        UB = 1
        Dim = 6
    elif F == 'F21':
        F_obj = F21
        LB = 0
        UB = 10
        Dim = 4
    elif F == 'F22':
        F_obj = F22
        LB = 0
        UB = 10
        Dim = 4
    elif F == 'F23':
        F_obj = F23
        LB = 0
        UB = 10
        Dim = 4
    return F_obj, LB, UB, Dim


# Individual functions
def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dim = len(x)
    o = 0
    for i in range(dim):
        o += np.sum(x[:i+1])**2
    return o

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    dim = len(x)
    return np.sum(100 * (x[1:dim] - x[:dim-1]**2)**2 + (x[:dim-1] - 1)**2)

def F6(x):
    return np.sum(np.abs(x + 0.5)**2)

def F7(x):
    dim = len(x)
    return np.sum(np.arange(1, dim+1) * (x**4)) + np.random.rand()

def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    dim = len(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dim

def F10(x):
    dim = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dim)) - np.exp(np.sum(np.cos(2 * np.pi * x)) / dim) + 20 + np.exp(1)

def F11(x):
    dim = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dim+1)))) + 1


def F12(x):
    dim = len(x)
    first_term = (np.pi / dim) * (
            10 * ((np.sin(np.pi * (1 + (x[0] + 1) / 4))) ** 2)
            + np.sum(((x[:dim - 1] + 1) / 4) ** 2 * (1 + 10 * (np.sin(np.pi * (1 + (x[1:dim] + 1) / 4))) ** 2))
    )
    second_term = ((x[dim - 1] + 1) / 4) ** 2
    ufun_term = np.sum(Ufun(x, 10, 100, 4))

    return first_term + second_term + ufun_term

def F13(x):
    dim = len(x)
    return 0.1 * ((np.sin(3 * np.pi * x[0]))**2 + np.sum((x[:dim-1] - 1)**2 * (1 + (np.sin(3 * np.pi * x[1:dim]))**2)) + (x[dim-1] - 1)**2 * (1 + (np.sin(2 * np.pi * x[dim-1]))**2)) + np.sum(Ufun(x, 5, 100, 4))

def F14(x):
    aS = np.array([
        [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
        [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
    ])
    bS = np.sum((x[:, np.newaxis] - aS)**6, axis=0)
    return (1 / 500 + np.sum(1 / (np.arange(1, 26) + bS)))**-1

def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    bK = 1 / bK
    return np.sum((aK - ((x[0] * (bK**2 + x[1] * bK)) / (bK**2 + 3 * bK + x[3])))**2)

def F16(x):
    return 4 * (x[0]**2) - 2.1 * (x[0]**4) + (x[0]**6) / 3 + x[0] * x[1] - 4 * (x[1]**2) + 4 * (x[1]**4)

def F17(x):
    return (x[1] - (x[0]**2) * 5.1 / (4 * (np.pi**2)) + 5 / np.pi * x[0] - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

def F18(x):
    return (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * (x[0]**2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * (x[1]**2))) * (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * (x[0]**2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1]**2)))

def F19(x):
    aH = np.array([
        [3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]
    ])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([
        [0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], 
        [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]
    ])
    
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return o

def F20(x):
    aH = np.array([
        [10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], 
        [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]
    ])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], 
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
    ])
    
    o = 0
    for i in range(4):
        o -= cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2)))
    return o

def F21(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], 
        [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
        [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    o = 0
    for i in range(5):
        o -= 1 / ((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i])  # Matrix multiplication for dot product
    return o

def F22(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], 
        [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
        [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    o = 0
    for i in range(7):
        o -= 1 / ((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i])  # Matrix multiplication for dot product
    return o

def F23(x):
    aSH = np.array([
        [4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], 
        [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
        [6, 2, 6, 2], [7, 3.6, 7, 3.6]
    ])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    
    o = 0
    for i in range(10):
        o -= 1 / ((x - aSH[i, :]) @ (x - aSH[i, :]).T + cSH[i])  # Matrix multiplication for dot product
    return o

def Ufun(x, a, k, m):
    return k * ((x - a)**m) * (x > a) + k * ((-x - a)**m) * (x < -a)