import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###________________________________________________________________###
# nhập hàm p(x)
def p(x):
    return 1+x**2
# nhập hàm q(x)
def q(x):
    return -(4*x**3 - 6*x**2 +2.25*x)
# nhập hàm f(x) (hoặc r(x))
def f(x):
    return 2*np.exp(-x**2)*(1.25*x**2-2.75*x-2)
# nhập a
a = 0
# nhập b
b = 3
# nhập n
n = 31
# nhập alpha (hoặc muy1)
alpha = 150
# nhập beta (hoặc muy2)
beta = 21.9787667
# nhập sigma1
sigma1 = 2
# nhập sigma2
sigma2 = 3

###_______________________________________________________________###

# giải bài toán điều kiện biên loại 1
def slove_DKBL1(a, b, n, alpha, beta):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    A = np.zeros((n, n))
    B = np.zeros(n)
    A[0][0] = 1
    A[n - 1][n - 1] = 1
    B[0] = alpha
    B[n - 1] = beta
    for i in range(1, n - 1):
        A[i][i - 1] = p(x[i] - h / 2)
        A[i][i] = -p(x[i] + h / 2) - p(x[i] - h / 2) + h**2 * q(x[i])
        A[i][i + 1] = p(x[i] + h / 2)
        B[i] = -h**2 * f(x[i])
    y = np.linalg.solve(A, B)
    return x, y

###_______________________________________________________________###
# giải bài toán điều kiện biên loại 2
def slove_DKBL2(a, b, n, muy1, muy2):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    A = np.zeros((n, n))
    B = np.zeros(n)
    A[0][0] = -p(x[0] + h / 2) + h**2 * q(x[0])/2
    A[0][1] = p(x[0] + h / 2)
    B[0] = -h**2 * f(x[0])/2 - muy1*h
    A[n - 1][n - 1] = p(x[n - 1] - h / 2) - h**2 * q(x[n - 1])/2
    A[n - 1][n - 2] = -p(x[n - 1] - h / 2)
    B[n - 1] = h**2 * f(x[n - 1])/2 - muy2*h
    for i in range(1, n - 1):
        A[i][i - 1] = p(x[i] - h / 2)
        A[i][i] = -p(x[i] + h / 2) - p(x[i] - h / 2) + h**2 * q(x[i])
        A[i][i + 1] = p(x[i] + h / 2)
        B[i] = -h**2 * f(x[i])
    y = np.linalg.solve(A, B)
    return x, y

###_______________________________________________________________###

# giải bài toán điều kiện biên loại 3
def slove_DKBL3(a, b, n, muy1, muy2, sigma1, sigma2):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    A = np.zeros((n, n))
    B = np.zeros(n)
    A[0][0] = -p(x[0] + h / 2) + h**2 * q(x[0])/2 - sigma1
    A[0][1] = p(x[0] + h / 2)
    B[0] = -h**2 * f(x[0])/2 - muy1*h
    A[n - 1][n - 1] = p(x[n - 1] - h / 2) - h**2 * q(x[n - 1])/2 - sigma2
    A[n - 1][n - 2] = -p(x[n - 1] - h / 2)
    B[n - 1] = h**2 * f(x[n - 1])/2 - muy2*h
    for i in range(1, n - 1):
        A[i][i - 1] = p(x[i] - h / 2)
        A[i][i] = -p(x[i] + h / 2) - p(x[i] - h / 2) + h**2 * q(x[i])
        A[i][i + 1] = p(x[i] + h / 2)
        B[i] = -h**2 * f(x[i])
    y = np.linalg.solve(A, B)
    return x, y

###______________________________________________________________________________###


# phương pháp sai phân với bài toán trị riêng
def sai_phan_tri_rieng(a, b, n, alpha, beta):
    h = (b - a) / (n - 1)
    x = np.linspace(a, b, n)
    A = np.zeros((n - 2, n - 2))
    A[0][0] = -p(x[1] + h / 2) / (f(x[1])*h**2) - p(x[1] - h / 2) / (f(x[1])*h**2) - q(x[1]) / f(x[1])
    A[0][1] = p(x[1] + h / 2) / (f(x[1])*h**2)
    A[n - 3][n - 3] = -p(x[n - 2] + h / 2) / (f(x[n - 2])*h**2) - p(x[n - 2] - h / 2) / (f(x[n - 2])*h**2) + q(x[n - 2]) / f(x[n - 2])
    A[n - 3][n - 4] = p(x[n - 2] - h / 2) / (f(x[n - 2])*h**2)
    for i in range(1, n - 3):
        A[i][i - 1] = p(x[i] - h / 2) / (f(x[i])*h**2)
        A[i][i] = -p(x[i] + h / 2) / (f(x[i])*h**2) - p(x[i] - h / 2) / (f(x[i])*h**2) - q(x[i]) / f(x[i])
        A[i][i + 1] = p(x[i] + h / 2) / (f(x[i])*h**2)
    lamda, y = np.linalg.eig(A)
    location = np.where(lamda == np.max(lamda))
    return lamda[location[0][0]], x, np.concatenate(([alpha], y[:, location[0][0]]/y[location[0][0], 0], [beta]))

###______________________________________________________________________________###

# hàm ghép trơn bậc 3
def ghep_tron_bac_3(x, y):
    S = list()
    h = list()
    m = list()
    lamda = list()
    muy = list()
    d = list()
    alpha = list()
    beta = list()
    phi = list()
    theta = list()

    lamda.append(1)
    muy.append(0)
    d.append(0)
    h.append(x[1] - x[0])

    alpha.append(0)
    beta.append(0)

    for i in range(0, len(x) - 2):
        h.append(x[i + 2] - x[i + 1])
        alpha.append(lamda[i] / (-2 - muy[i] * alpha[i]))
        beta.append((beta[i]*muy[i] - d[i])/(-2 - muy[i] * alpha[i]))
        lamda.append(h[i+1]/(h[i] + h[i+1]))
        muy.append(1 - lamda[i+1])
        d.append(6*((y[i+2] - y[i+1])/h[i+1] - (y[i+1] - y[i])/h[i])/(h[i] + h[i+1]))

    alpha.append(lamda[-1]/(-2 - muy[-1]*alpha[-1]))
    beta.append((beta[-1]*muy[-1] - d[-1])/(-2 - muy[-1]*alpha[-2]))

    lamda.append(0)
    muy.append(1)
    d.append(0)

    m.append((muy[-1]*beta[-1] - d[-1])/(-2 - muy[-1]*alpha[-1]))

    for i in range(len(x) - 2, -1, -1):
        m.append(alpha[i+1]*m[-1] + beta[i+1])
        phi.append(y[i+1]/h[i] - m[-2]*h[i]/6)
        theta.append(y[i]/h[i] - m[-1]*h[i]/6)
        s = list()
        s.append((-m[-1] + m[-2])/(6*h[i]))
        s.append((m[-1]*x[i+1] - m[-2]*x[i])/(2*h[i]))
        s.append((-m[-1]*x[i+1]**2 + m[-2]*x[i]**2)/(2*h[i]) + phi[-1] - theta[-1])
        s.append((m[-1]*x[i+1]**3 - m[-2]*x[i]**3)/(6*h[i]) - phi[-1]*x[i] + theta[-1]*x[i+1])
        S.append(np.array(s))
    S.reverse()
    return S

def hoocne_quatient(a, x):
    # chia gia tri cua da thuc P(x) cho (x - x_0)
    # tra ve b và b_0 trong do:
    # b la he so cua da thuc sau khi chia
    # b_0 la phan du va la ket qua cua P(x)
    y = list()
    y.append(a[0])
    for i in range(len(a) - 1):
        y.append(y[i] * x + a[i + 1])
    b = np.array(y[:-1])
    b_0 = np.array(y[-1])
    return b, b_0

# vẽ đồ thị
def ve_do_thi(x, y, S):
    # plt.plot(x, y, 'ro')
    for i in range(len(S)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S[i], x_0)[1]
        plt.plot(x_0, y_0, 'b')
    plt.show()

def ve_do_thi_2(x, y, S1, S2):
    # plt.plot(x, y, 'ro')
    for i in range(len(S1)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S1[i], x_0)[1]
        Y0 = plt.plot(x_0, y_0, 'r')
    for i in range(len(S2)):
        x_1 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_1 = hoocne_quatient(S2[i], x_1)[1]
        Y1 = plt.plot(x_1, y_1, 'g')
    plt.legend([Y0[0], Y1[0]], ['y[0]: biến thứ nhất', 'y[1]: biến thứ hai'])
    plt.show()

    plt.plot(y[:, 0], y[:, 1], 'y')
    plt.xlabel('y[0]')
    plt.ylabel('y[1]')
    plt.show()

def ve_do_thi_3(x, y, S1, S2, S3):
    # plt.plot(x, y, 'ro')
    for i in range(len(S1)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S1[i], x_0)[1]
        Y0 = plt.plot(x_0, y_0, 'r')
    for i in range(len(S2)):
        x_1 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_1 = hoocne_quatient(S2[i], x_1)[1]
        Y1 = plt.plot(x_1, y_1, 'g')
    for i in range(len(S3)):
        x_2 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_2 = hoocne_quatient(S2[i], x_2)[1]
        Y2 = plt.plot(x_2, y_2, 'b')
    plt.legend([Y0[0], Y1[0], Y2[0]], ['y[0]', 'y[1]', 'y[2]'])
    plt.show()

    plt.plot(y[:, 0], y[:, 1], 'y')
    plt.xlabel('y[0]')
    plt.ylabel('y[1]')
    plt.show()

    plt.plot(y[:, 0], y[:, 2], 'y')
    plt.xlabel('y[0]')
    plt.ylabel('y[2]')
    plt.show()

    plt.plot(y[:, 1], y[:, 2], 'y')
    plt.xlabel('y[1]')
    plt.ylabel('y[2]')

###___________________________________________________________________###
def Giaibienloai1():
    # giải bài toán điều kiện biên loại 1
    print("Bài toán điều kiện biên loại 1")
    x, y = slove_DKBL1(a, b, n, alpha, beta)
    print(np.min(y))
    print(np.concatenate((x, y)).reshape(2, n).T)
    S = ghep_tron_bac_3(x, y)
    ve_do_thi(x, y, S)

###___________________________________________________________________###
def Giaibienloai2():
    # giải bài toán điều kiện biên loại 2
    print("Bài toán điều kiện biên loại 2")
    x, y = slove_DKBL2(a, b, n, alpha, beta)
    print(np.concatenate((x, y)).reshape(2, n).T)

###___________________________________________________________________###
def Giaibienloai3():
    # giải bài toán điều kiện biên loại 3
    print("Bài toán điều kiện biên loại 3")
    if sigma1 >= 0 and sigma2 >= 0 and sigma1 + sigma2 > 0:
        x, y = slove_DKBL3(a, b, n, alpha, beta, sigma1, sigma2)
        print(np.concatenate((x, y)).reshape(2, n).T)

###___________________________________________________________________###

def Giai_Tri_Rieng():
    # phương pháp sai phân với bài toán trị riêng
    print("Phương pháp sai phân với bài toán trị riêng")
    lamda, x, y = sai_phan_tri_rieng(a, b, n, alpha, beta)
    print("Trị riêng trội là: ", lamda)
    print("bảng giá trị của hàm số tương ứng là: ")
    print(np.concatenate((x, y)).reshape(2, n).T)

choose = int(input("Chọn bài toán biên loại k (k = 1,2,3 và 4 là BT trị riêng) : "))
if choose == 1:
    Giaibienloai1()
if choose == 2:
    Giaibienloai2()
if choose == 3:
    Giaibienloai3()
if choose == 4:
    Giai_Tri_Rieng()
