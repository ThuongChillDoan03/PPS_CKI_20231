import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


###_________________________________________________________________________###
# nhập hàm f(x, y) = y'
def f(x, y):    
    return np.array([x + y[0]])     ###Thay đổi hàm ở đây

# vị trí đầu
a = 0
# vị trí cuối
b = 0.5
# số mốc
n = 6
# giá trị y0
y0 = np.array([1])
# sai số
eps = 1e-4

###_________________________________________________________________________###


# tính bằng euler hiện
def Euler_hien(a, b, y0, n):
    h = (b - a) / (n-1)
    x = np.linspace(a, b, n)
    y = [y0]
    
    for i in range(n - 1):
        y.append(y[i] + h * f(x[i], y[i]))

    return x, np.array(y)


# tính bằng euler ẩn
def Euler_an(a, b, y0, n, eps):
    h = (b - a) / (n-1)
    x = np.linspace(a, b, n)
    y = [y0]
    for i in range(n - 1):
        y.append(y[i] + h * f(x[i], y[i]))
        print("------------------")
        print("x = ", x[i + 1], "y = ", y[i + 1])
        yTerm = np.copy(y[i+1] + 1)
        k = 1
        while np.linalg.norm(yTerm - y[i+1]) > eps and k < 1000:
            print("lần lặp thứ ", k)
            k += 1
            yTerm = np.copy(y[i+1])
            y[i+1] = y[i] + h * f(x[i + 1], y[i + 1])
            print("y = ", y[i+1])
        if k > 990:
            print("=====================================")
            print("Số vòng lặp quá lớn !!!")
            print("=====================================")
            return 0, 0
    return x, np.array(y)

# tính bằng công thức hình thang
def hinh_thang(a, b, y0, n, eps):
    h = (b - a) / (n-1)
    x = np.linspace(a, b, n)
    y = [y0]

    for i in range(n - 1):
        y.append(y[i] + h * f(x[i], y[i]))
        print("------------------")
        print("x = ", x[i + 1], "y = ", y[i + 1])
        yTerm = np.copy(y[i+1] + 1)
        k = 1
        while np.linalg.norm(yTerm - y[i+1]) > eps:
            print("lần lặp thứ ", k)
            k += 1
            yTerm = np.copy(y[i+1])
            y[i+1] = y[i] + h/2 * (f(x[i], y[i]) + f(x[i+1], y[i+1]))
            print("y = ", y[i+1])
        
    return x, np.array(y)

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
    plt.legend([Y0[0], Y1[0]], ['y[0]', 'y[1]'])
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


bac = y0.shape[0]

###________________________________________________________________________________###
print("Euler hiện:")
print("=====================================")
x, y = Euler_hien(a, b, y0, n)
if bac == 1:
    print(np.concatenate((x, y[:, 0])).reshape(2, n).T)
    print("=====================================")
    S = ghep_tron_bac_3(x, y[:, 0])
    ve_do_thi(x, y, S)
elif bac == 2:
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1])))).reshape(3, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    ve_do_thi_2(x, y, S1, S2)
elif bac == 3:
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1], y[:, 2])))).reshape(4, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    S3 = ghep_tron_bac_3(x, y[:, 2])
    ve_do_thi_3(x, y, S1, S2, S3)
# print("=====================================")
# while True:
#     x_0 = float(input("Nhập x_0: "))
#     location = 0
#     for i in range(x.shape[0] - 1):
#         if x[i] <= x_0 and x_0 <= x[i + 1]:
#             location = i
#             break
#     print("y({}) = ".format(x_0), hoocne_quatient(S[location], x_0)[1])


###________________________________________________________________________________###
print("Euler ẩn:")
x, y = Euler_an(a, b, y0, n, eps)
print("=====================================")
if bac == 1:
    print(np.concatenate((x, y[:, 0])).reshape(2, n).T)
    print("=====================================")
    S = ghep_tron_bac_3(x, y[:, 0])
    ve_do_thi(x, y, S)
elif bac == 2:
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1])))).reshape(3, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    ve_do_thi_2(x, y, S1, S2)
elif bac == 3:
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1], y[:, 2])))).reshape(4, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    S3 = ghep_tron_bac_3(x, y[:, 2])
    ve_do_thi_3(x, y, S1, S2, S3)
# print("=====================================")
# while True:
#     x_0 = float(input("Nhập x_0: "))
#     location = 0
#     for i in range(x.shape[0] - 1):
#         if x[i] <= x_0 and x_0 <= x[i + 1]:
#             location = i
#             break
#     print("y({}) = ".format(x_0), hoocne_quatient(S[location], x_0)[1])


print("Hình thang:")
x, y = hinh_thang(a, b, y0, n, eps)
print("=====================================")
if bac == 1:
    print(np.concatenate((x, y[:, 0])).reshape(2, n).T)
    print("=====================================")
    S = ghep_tron_bac_3(x, y[:, 0])
    ve_do_thi(x, y, S)
elif bac == 2:
    print(y[: : 2, 0])
    print(y[: : 2, 1])
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1])))).reshape(3, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    ve_do_thi_2(x, y, S1, S2)
elif bac == 3:
    print(np.concatenate((x, np.concatenate((y[:, 0], y[:, 1], y[:, 2])))).reshape(4, n).T)
    S1 = ghep_tron_bac_3(x, y[:, 0])
    S2 = ghep_tron_bac_3(x, y[:, 1])
    S3 = ghep_tron_bac_3(x, y[:, 2])
    ve_do_thi_3(x, y, S1, S2, S3)
# print("=====================================")
# while True:
#     x_0 = float(input("Nhập x_0: "))
#     location = 0
#     for i in range(x.shape[0] - 1):
#         if x[i] <= x_0 and x_0 <= x[i + 1]:
#             location = i
#             break
#     print("y({}) = ".format(x_0), hoocne_quatient(S[location], x_0)[1])
    


