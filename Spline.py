import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def doc_input (ten_file):
    print("======================================================")
    print("Kieu du lieu dau vao: ")
    print("1. Ngang")
    print("2. Doc")
    chon = int(input("chon kieu dau vao: "))
    if(chon == 1):
        inp = pd.read_excel(ten_file)
        b = np.asarray(inp.astype(np.float64))
        # doc du lieu cua x va y
        x = b[0]
        y = b[1]
    else:
        inp = pd.read_excel(ten_file)
        b = np.asarray(inp.astype(np.float64))
        b = b.T
        x = b[0]
        y = b[1]
    return x, y

def kiem_tra_input (x, y):
    #tra ve 1 khi input hop le va 0 khi input khong hop le
    # kiem tra kich thuoc du lieu
    if (x.shape[0] != y.shape[0]):
        print("kich thuoc khong hop le")
        return 0
    
    # kiem tra du lieu trung
    for i in x:
        if (np.where(x == i)[0].shape[0] > 1):
            print("du lieu cua x o cac vi tri ", np.where(x == i)[0], " trung nhau")
            return 0
    # input hop le
    print("input hop le")
    return 1

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


# hàm ghép trơn bậc 2
def ghep_tron_bac_2(x, y):
    S = list()
    m = list()
    h = list()

    m.append((y[1] - y[0]) / (x[1] - x[0]))

    for i in range(0, len(x) - 1):
        h.append(x[i + 1] - x[i])
        m.append(-m[i] + 2/h[i]*(y[i + 1] - y[i]))
        s = list()
        s.append((m[i + 1] - m[i]) / (2 * h[i]))
        s.append((m[i] * x[i + 1] - m[i + 1] * x[i]) / h[i])
        s.append((-m[i]*x[i+1]**2 + m[i+1]*x[i]**2)/(2*h[i]) + y[i] + m[i]*h[i]/2)
        S.append(np.array(s))

    return S

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


def ve_do_thi1(x, y, S2):
    plt.plot(x, y, 'ro')
    for i in range(len(S2)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S2[i], x_0)[1]
        plt.plot(x_0, y_0, 'b')
    plt.show()

def ve_do_thi2(x, y, S3):
    plt.plot(x, y, 'ro')
    for i in range(len(S3)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S3[i], x_0)[1]
        plt.plot(x_0, y_0, 'g')
    plt.show()

def ve_do_thi1_2(x, y, S2, S3):
    plt.plot(x, y, 'ro')
    for i in range(len(S2)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S2[i], x_0)[1]
        plt.plot(x_0, y_0, 'b')
    
    for i in range(len(S3)):
        x_0 = np.linspace(x[i], x[i + 1], int((x[i + 1] - x[i])*1000))
        y_0 = hoocne_quatient(S3[i], x_0)[1]
        plt.plot(x_0, y_0, 'g')
    plt.show()

x, y = doc_input("Data_Spline.xlsx")
if (kiem_tra_input(x, y) == 1):
    S2 = ghep_tron_bac_2(x, y)
    print("====================================================================================================")
    print("HÀM GHÉP TRƠN BẬC 2")
    print("da thuc S2(x) = ")
    for i in range(len(S2) - 1):
        print("S[{}, {}](x) = {}x^2 + {}x + {}".format(x[i], x[i + 1], S2[i][0], S2[i][1], S2[i][2]))
    print("S[{}, {}](x) = {}x^2 {}x + {}".format(x[-2], x[-1], S2[-1][0], S2[-1][1], S2[-1][2]))
    ve_do_thi1(x, y, S2)
    
    S3 = ghep_tron_bac_3(x, y)
    print("====================================================================================================")
    print("HÀM GHÉP TRƠN BẬC 3")
    print("da thuc S3(x) = ")
    for i in range(len(S3) - 1):
        print("S[{}, {}](x) = {}x^3 + {}x^2 + {}x + {}".format(x[i], x[i + 1], S3[i][0], S3[i][1], S3[i][2], S3[i][3]))
    print("S[{}, {}](x) = {}x^3 + {}x^2 {}x + {}".format(x[-2], x[-1], S3[-1][0], S3[-1][1], S3[-1][2], S3[-1][3]))
    print(ve_do_thi1(x, y, S3))


    ve_do_thi1_2(x, y, S2, S3)

    while(True):
        print("====================================================================================================")
        print("Nhap vao x de tinh gia tri cua S2(x) va S3(x)")
        x_0 = float(input())
        location = 0
        for i in range(len(x) - 1):
            if x[i] <= x_0 and x_0 <= x[i + 1]:
                location = i
                break
        print(S2[location])
        print(S3[location])
        print("S2({}) = {}".format(x_0, hoocne_quatient(S2[location], x_0)[1]))
        print("S3({}) = {}".format(x_0, hoocne_quatient(S3[location], x_0)[1]))
else:
    print("Input khong hop le")



