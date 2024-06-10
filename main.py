import numpy as np
import matplotlib.pyplot as plt

p1 = 8.4E-6
p2 = 6.6667E-4
p3 = 1.7778E-5
p4 = 0.01
p5 = 2
p6 = 10


def JacobyMatrix(x1, x2, x3, p4):
    return np.array([
        [(1 - 2 * x1 - x2) / p2 - p4, (p1 - x1) / p2, 0],
        [-x2 / p3, -(p1 + x1) / p3 - p4, p5 / p6],
        [1, 0, -1 - p4]
    ])


def check(x1, x2, x3, p4):
    flag = True
    for i in range(3):
        if np.round((p1 * x2[i] - x1[i] * x2[i] + x1[i] - x1[i] * x1[i]) / p2 - p4 * x1[i], 9) != 0:
            flag = False
    for i in range(3):
        if np.round((-p1 * x2[i] - x1[i] * x2[i] + p5 * x3[i]) / p3 + p4 * (p6 - x2[i]), 9) != 0:
            flag = False
    for i in range(3):
        if np.round((x1[i] - x3[i] - p4 * x3[i]), 9) != 0:
            flag = False
    if not flag:
        print(x1, x2, x3, flag)
        exit(-1)


def calculate_eigenvalues(p4, p1, p2, p3, p5, p6):
    x1Array1, x2Array1, x3Array1 = [], [], []
    x1Array2, x2Array2, x3Array2 = [], [], []
    x1Array3, x2Array3, x3Array3 = [], [], []
    p4Array = []
    eigenvalues1, eigenvalues2, eigenvalues3 = [], [], []

    while p4 < 1000:
        x1 = [-(1 + p4),
              (1 + p4) * (-p3 * p4 - p1 + 1 - p2 * p4) - p5,
              (1 + p4) * (p3 * p4 - p2 * p3 * p4 ** 2 + p1 - p1 * p2 * p4 - p3 * p4 * p6) + p1 * p5,
              p1 * p3 * p4 * p6 * (1 + p4)]
        x1 = np.roots(x1)
        for i in range(3):
            if x1[i] == p1:
                print(p4, x1)
        x1 = [root for root in x1 if root != p1]

        x2 = [((x1[0]) ** 2 + (p2 * p4 - 1) * x1[0]) / (p1 - x1[0]),
              ((x1[1]) ** 2 + (p2 * p4 - 1) * x1[1]) / (p1 - x1[1]),
              ((x1[2]) ** 2 + (p2 * p4 - 1) * x1[2]) / (p1 - x1[2])]
        x3 = [x1[0] / (1 + p4), x1[1] / (1 + p4), x1[2] / (1 + p4)]

        check(x1, x2, x3, p4)

        jacobian = JacobyMatrix(x1[0], x2[0], x3[0], p4)
        eigenvalues = np.linalg.eigvals(jacobian)
        eigenvalues1.append(eigenvalues)

        jacobian = JacobyMatrix(x1[1], x2[1], x3[1], p4)
        eigenvalues = np.linalg.eigvals(jacobian)
        eigenvalues2.append(eigenvalues)

        jacobian = JacobyMatrix(x1[2], x2[2], x3[2], p4)
        eigenvalues = np.linalg.eigvals(jacobian)
        eigenvalues3.append(eigenvalues)

        p4Array.append("{:.2f}".format(round(p4, 2)))
        x1Array1.append(x1[0])
        x2Array1.append(x2[0])
        x3Array1.append(x3[0])
        x1Array2.append(x1[1])
        x2Array2.append(x2[1])
        x3Array2.append(x3[1])
        x1Array3.append(x1[2])
        x2Array3.append(x2[2])
        x3Array3.append(x3[2])

        if p4 < 0.1:
            p4 += 0.01
        elif p4 < 1:
            p4 += 0.1
        elif p4 < 10:
            p4 += 1
        elif p4 < 100:
            p4 += 10
        else:
            p4 += 100

    np.savetxt('x1Array1.txt', x1Array1)
    np.savetxt('x2Array1.txt', x2Array1)
    np.savetxt('x3Array1.txt', x3Array1)
    np.savetxt('x1Array2.txt', x1Array2)
    np.savetxt('x2Array2.txt', x2Array2)
    np.savetxt('x3Array2.txt', x3Array2)
    np.savetxt('x1Array3.txt', x1Array3)
    np.savetxt('x2Array3.txt', x2Array3)
    np.savetxt('x3Array3.txt', x3Array3)
    # Convert p4Array to a numerical type before saving
    p4Array_numeric = np.array(p4Array, dtype=float)
    np.savetxt('p4Array.txt', p4Array_numeric)

    # Convert eigenvalues to a numerical type before saving
    eigenvalues1_numeric = np.array(eigenvalues1)
    np.savetxt('eigenvalues1.txt', eigenvalues1_numeric)

    eigenvalues2_numeric = np.array(eigenvalues2)
    np.savetxt('eigenvalues2.txt', eigenvalues2_numeric)

    eigenvalues3_numeric = np.array(eigenvalues3)
    np.savetxt('eigenvalues3.txt', eigenvalues3_numeric)

    # X1
    plt.subplot(1, 3, 1)  # 1 строка, 3 столбца, первый график
    for i in range(len(x1Array1)):
        if np.all(np.real(eigenvalues1[i]) < 0):
            plt.plot(x1Array1[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x1Array1[i], p4Array[i], 'ro', ms=1)
    plt.title('X1[1]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 2)  # 1 строка, 3 столбца, первый график
    for i in range(len(x1Array2)):
        if np.all(np.real(eigenvalues2[i]) < 0):
            plt.plot(x1Array2[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x1Array2[i], p4Array[i], 'ro', ms=1)
    plt.title('X1[2]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 3)
    for i in range(len(x1Array3)):
        if np.all(np.real(eigenvalues3[i]) < 0):
            plt.plot(x1Array3[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x1Array3[i], p4Array[i], 'ro', ms=1)
    plt.title('X1[3]')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('X1, p6 =' + str(p6), fontsize=16)
    plt.tight_layout()
    plt.show()

    # X2
    plt.subplot(1, 3, 1)
    for i in range(len(x2Array1)):
        if np.all(np.real(eigenvalues1[i]) < 0):
            plt.plot(x2Array1[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x2Array1[i], p4Array[i], 'ro', ms=1)
    plt.title('X2[1]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 2)
    for i in range(len(x2Array2)):
        if np.all(np.real(eigenvalues2[i]) < 0):
            plt.plot(x2Array2[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x2Array2[i], p4Array[i], 'ro', ms=1)
    plt.title('X2[2]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 3)
    for i in range(len(x2Array3)):
        if np.all(np.real(eigenvalues3[i]) < 0):
            plt.plot(x2Array3[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x2Array3[i], p4Array[i], 'ro', ms=1)
    plt.title('X2[3]')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('X2, p6 =' + str(p6), fontsize=16)
    plt.tight_layout()
    plt.show()

    # X3
    plt.subplot(1, 3, 1)
    for i in range(len(x3Array1)):
        if np.all(np.real(eigenvalues1[i]) < 0):
            plt.plot(x3Array1[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x3Array1[i], p4Array[i], 'ro', ms=1)
    plt.title('X3[1]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 2)
    for i in range(len(x3Array2)):
        if np.all(np.real(eigenvalues2[i]) < 0):
            plt.plot(x3Array2[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x3Array2[i], p4Array[i], 'ro', ms=1)
    plt.title('X3[2]')
    plt.xscale('log')
    plt.yscale('log')

    plt.subplot(1, 3, 3)
    for i in range(len(x3Array3)):
        if np.all(np.real(eigenvalues3[i]) < 0):
            plt.plot(x3Array3[i], p4Array[i], 'go', ms=1)
        else:
            plt.plot(x3Array3[i], p4Array[i], 'ro', ms=1)
    plt.title('X3[3]')
    plt.xscale('log')
    plt.yscale('log')

    plt.suptitle('X3, p6 =' + str(p6), fontsize=16)
    plt.tight_layout()
    plt.show()


calculate_eigenvalues(p4, p1, p2, p3, p5, p6)

p4Array = []
p6 = 100

calculate_eigenvalues(p4, p1, p2, p3, p5, p6)
