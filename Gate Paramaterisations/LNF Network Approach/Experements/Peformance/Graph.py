import matplotlib.pyplot as plt
import numpy as np
import glob
files = glob.glob('./results/*.txt')

def conf_interval(data):
    N = len(data)
    M = data.mean()
    sorted_estimates = np.sort(np.array(data))
    conf_interval = (np.abs(sorted_estimates[int(0.025 * N)] - M), np.abs(sorted_estimates[int(0.975 * N)] - M))
    return conf_interval

def plot(x, y, ci, c, name):
    ci=np.transpose(np.array(ci))
    plt.errorbar(x, y, yerr=ci, marker='o', color=c, label=name)

cnf_peformance = {}
dnf_peformance = {}
pcep_peformance = {}
pcep_g_peformance = {}

for file in files:
    file_comp = file.split('-')
    n = int(file_comp[1])
    
    f = open(file)
    print(file)
    sec = f.readline().strip().split(":")
    print(sec)
    f.close()

    
    cnf_er = float(sec[0])
    dnf_er = float(sec[1])
    pcep_er = float(sec[2])
    pcep_g_er = float(sec[3])

    

    cnf_peformance.setdefault(n, [])
    cnf_peformance[n].append(cnf_er)
    dnf_peformance.setdefault(n, [])
    dnf_peformance[n].append(dnf_er)
    pcep_peformance.setdefault(n, [])
    pcep_peformance[n].append(pcep_er)
    pcep_g_peformance.setdefault(n, [])
    pcep_g_peformance[n].append(pcep_g_er)


x_axis = np.array(range(2, 10))
cnf_ci = []
dnf_ci = []
pcep_ci = []
pcep_g_ci = []

cnf_data = []
dnf_data = []
pcep_data = []
pcep_g_data = []

for i in x_axis:
    cnf_pef = np.array(cnf_peformance[i])
    dnf_pef = np.array(dnf_peformance[i])
    pcep_pef = np.array(pcep_peformance[i])
    pcep_g_pef = np.array(pcep_g_peformance[i])

    cnf_data.append(cnf_pef.mean())
    cnf_ci.append(conf_interval(cnf_pef))

    dnf_data.append(dnf_pef.mean())
    dnf_ci.append(conf_interval(dnf_pef))

    pcep_data.append(pcep_pef.mean())
    pcep_ci.append(conf_interval(pcep_pef))

    pcep_g_data.append(pcep_g_pef.mean())
    pcep_g_ci.append(conf_interval(pcep_g_pef))

plot(x_axis, cnf_data, cnf_ci, 'b', 'CNF')
plot(x_axis, dnf_data, dnf_ci, 'r', 'DNF')
plot(x_axis, pcep_data, pcep_ci, 'g', 'Perceptron (Same Config)')
plot(x_axis, pcep_g_data, pcep_g_ci, 'y', 'Perceptron')

plt.ylabel("Error Rate")
plt.xlabel("Size Of Expression")
plt.xlim([2-1, 10 + 1])
plt.legend(loc='best')
plt.savefig("all-peformance.png")

plt.clf()

plot(x_axis, cnf_data, cnf_ci, 'b', 'CNF')
plot(x_axis, dnf_data, dnf_ci, 'r', 'DNF')

plt.ylabel("Error Rate")
plt.xlabel("Size Of Expression")
plt.xlim([2-1, 10 + 1])
plt.legend(loc='best')
plt.savefig("lnfn-peformance.png")
