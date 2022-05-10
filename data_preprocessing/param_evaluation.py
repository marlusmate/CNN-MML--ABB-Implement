import matplotlib.pyplot as plt
import numpy as np
import pickle

# Path to data
data_all_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/02_data'
# Path to data
data_exp_folder = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/02_raw_data'

# Dataset parameters
exp_list = ["exp_2022-02-25",
            "exp_2022-03-09"]
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level", "flow_regime"]
no_classes = 3
split_ratio = [0.9, 0.1, 0.0]
output_proc_shape = (len(param_list),)
output_img_shape = (128, 128, 1)

# Def Functions


def froude(rpm, d=0.3, g=9.81):
    fr = (pow(rpm/60,2)*d) / g

    return fr


def flow(rpm, gfl, d=0.3):
    fl = (gfl*pow(10,-3)) / (rpm * pow(d,3))

    return fl


# Load alle Process param points
with open('data/data_points_all.pickle', 'rb') as f:
    data = pickle.load(f)
    print("Data geladen")

params = [d[:4] for d in data]
label = [d[4] for d in data]
rpm = [d[0] for d in data]
gfl = [d[1] for d in data]
temp = [d[2] for d in data]

froude_ = [froude(rpm_i) for rpm_i in rpm]
flow_ = [flow(rpm_i, gfl_i) for rpm_i, gfl_i, in zip(rpm, gfl)]
print(data[0])

ax1 = plt.subplot(131)
ax1.boxplot(rpm)
ax1.set_title('Rührerdrehzahl [1/min]')
ax1.set_xticks(ticks=[0], label=None)
ax2 = plt.subplot(132)
ax2.boxplot(gfl)
ax2.set_title('Begasungsrate [l/min]')
ax2.set_xticks(ticks=[0], label=None)
ax3 = plt.subplot(133)
ax3.boxplot(temp)
ax3.set_title('Temperatur [°C]')
ax3.set_xticks(ticks=[0], label=None)

# Class Fine params
rpm_0 = [rpm[i] for i in np.arange(len(rpm)) if label[i] == 0]
rpm_1 = [rpm[i] for i in np.arange(len(rpm)) if label[i] == 1]
rpm_2 = [rpm[i] for i in np.arange(len(rpm)) if label[i] == 2]

gfl_0 = [gfl[i] for i in np.arange(len(gfl)) if label[i] == 0]
gfl_1 = [gfl[i] for i in np.arange(len(gfl)) if label[i] == 1]
gfl_2 = [gfl[i] for i in np.arange(len(gfl)) if label[i] == 2]

temp_0 = [temp[i] for i in np.arange(len(temp)) if label[i] == 0]
temp_1 = [temp[i] for i in np.arange(len(temp)) if label[i] == 1]
temp_2 = [temp[i] for i in np.arange(len(temp)) if label[i] == 2]


plt.figure(1)
plt.plot(temp_0, rpm_0, 'b*')
plt.plot(temp_1, rpm_1, 'm*')
plt.plot(temp_2, rpm_2, 'g*')
plt.xlabel("Temperatur [°C]")
plt.ylabel("Rührerdrehzahl [1/min]")
plt.title("Scatterplot  Temperatur Rührerdrehzahl")
plt.legend(["flooded", "loaded", "dispersed"])

plt.figure(2)
plt.plot(gfl_0, rpm_0, 'b*')
plt.plot(gfl_1, rpm_1, 'm*')
plt.plot(gfl_2, rpm_2, 'g*')
plt.xlabel("Begasungsrate [l/min]")
plt.ylabel("Rührerdrehzahl [1/min]")
plt.title("Scatterplot  Begasungsrate Rührerdrehzahl")
plt.legend(["flooded", "loaded", "dispersed"])