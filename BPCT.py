import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#nhập các hàm phi
def phi(x):
    return [x, x**2]


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

def M(x):
    Phi = list()
    for i in range(len(x)):
        Phi.append(phi(x[i]))
    return np.array(Phi)

# tính hệ số bằng bình phương cực tiểu
def bpcu_tieu(x, y):
    Phi = M(x)
    Phi_T = Phi.T
    a = np.dot(Phi_T, Phi)
    b = np.dot(Phi_T, y)
    print("======================================================")
    print("Ma tran he so M^T.M: \n", a)
    print("======================================================")
    print("Ma tran he so M^T.y: \n", b)
    print("======================================================")
    print("Ma tran nghịch đảo M^T: \n", np.linalg.inv(a))
    print("======================================================")
    return np.linalg.solve(a, b)



# trường hợp ham y = a*exp(b1*phi1(x) + b2*phi2(x) + ...)
def ham_mu(x, y):
    if(np.all(y > 0)):
        return bpcu_tieu(x, np.log(y)), 1
    elif(np.all(y < 0)):
        return bpcu_tieu(x, np.log(-y)), -1
    else:
        print("khong the tinh")
        return 0
    

# trường hợp ham y = a*x^b
def ham_luy_thua(x, y):
    if(np.all(y > 0) and np.all(x > 0)):
        return bpcu_tieu(x, np.log(y)), 1
    elif(np.all(y < 0) and np.all(x > 0)):
        return bpcu_tieu(x, np.log(-y)), -1
    else:
        print("khong the tinh")
        return 0
    

# trường hợp ham y = ln(a1*phi1(x) + a2*phi2(x) + ...)
def ham_ln(x, y):
    return bpcu_tieu(x, np.exp(y))


# vẽ đồ thị
def ve_do_thi(x, y, a, kieu, sgn):
    plt.plot(x, y, 'ro')
    # vẽ đường thẳng
    x1 = np.linspace(x[0], x[-1], int(100*(x[-1] - x[0])))
    if (kieu == 1):
        y1 = np.exp(np.dot(M(x1), a))*sgn
    elif (kieu == 2):
        y1 = np.exp(np.dot(M(x1), a))*sgn
    elif (kieu == 3):
        y1 = np.log(np.dot(M(x1), a))
    else:
        y1 = np.dot(M(x1), a)
    plt.plot(x1, y1, 'b-')
    plt.show()


# chay chuong trinh
x, y = doc_input("Data_2_BPCT.xlsx")
if (kiem_tra_input(x, y) == 1):
    print("======================================================")
    print("Kieu ham so: ")
    print("1. y = a*exp(b1*phi1(x) + b2*phi2(x) + ...)")
    print("2. y = a*x^b")
    print("3. y = ln(a1*phi1(x) + a2*phi2(x) + ...)")
    print("4. y = a1*phi1(x) + a2*phi2(x) + ... + an*phin(x)")
    chon = int(input("chon kieu ham so: "))
    if(chon == 1):
        a, sgn = ham_mu(x, y)
        print("======================================================")
        print("He so cua ham so: ")
        print("a = ", np.exp(a[0])*sgn)
        print("b = ", a[1:])
        print("======================================================")
        print("Sai so: {}".format(np.sqrt(np.sum((y - np.exp(np.dot(M(x), a))*sgn)**2) / x.shape[0])))
        ve_do_thi(x, y, a, chon, sgn)
        while(True):
            x = float(input("Nhap gia tri x can tinh: "))
            print("y = {}".format(np.exp(np.dot(phi(x), a))*sgn))
    elif(chon == 2):
        a, sgn = ham_luy_thua(x, y)
        print("======================================================")
        print("He so cua ham so: ")
        print("a = ", np.exp(a[0])*sgn)
        print("b = ", a[1])
        print("======================================================")
        print("Sai so: {}".format(np.sqrt(np.sum((y - np.exp(np.dot(M(x), a))*sgn)**2) / x.shape[0])))
        ve_do_thi(x, y, a, chon, sgn)
        while(True):
            x = float(input("Nhap gia tri x can tinh: "))
            print("y = {}".format(np.exp(np.dot(phi(x), a))*sgn))
    elif(chon == 3):
        a = ham_ln(x, y)
        print("======================================================")
        print("He so cua ham so: ")
        print("a = ", a)
        print("======================================================")
        print("Sai so: {}".format(np.sqrt(np.sum((y - np.log(np.dot(M(x), a)))**2) / x.shape[0])))
        ve_do_thi(x, y, a, chon, 0)
        while(True):
            x = float(input("Nhap gia tri x can tinh: "))
            print("y = {}".format(np.log(np.dot(phi(x), a))))
    elif(chon == 4):
        a = bpcu_tieu(x, y)
        print("======================================================")
        print("He so cua ham so: ")
        print("a = ", a)
        print("======================================================")
        print("Sai so: {}".format(np.sqrt(np.sum((y - np.dot(M(x), a))**2) / x.shape[0])))
        ve_do_thi(x, y, a, chon, 0)
        while(True):
            x = float(input("Nhap gia tri x can tinh: "))
            print("y({}) = {}".format(x, np.dot(phi(x), a)))
    else:
        print("khong co lua chon nay")


