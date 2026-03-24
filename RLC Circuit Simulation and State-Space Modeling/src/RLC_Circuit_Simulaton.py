import numpy as np #type: ignore
import matplotlib.pyplot as plt #type: ignore

def store(M, filename):
    M.tofile(filename + ".bin")

# constants
T = 1e-3    # sampling period (s)
t_dur = 0.2 # duration of simulation (s)
R = 1.0     # resistance (ohm)
L = 10e-3   # inductance (H)
C = 10e-3   # capacitance (F)

# time array
K = round(t_dur/T) + 1  # points in time
t = np.arange(K)*T

# states and inputs
x = np.zeros(2) # iL (A), vC (V)
vs = 0          # source voltage (V)

# storage 
x_t = np.zeros((2, K))
i_t = np.zeros((2, K))
r_t = np.zeros(K)

for k in range(K):

    # updates
    iR = x[1]/R
    iC = x[0] - iR
    if x[0]==0:
        Rp = np.nan
    else:
        Rp = x[1] / x[0]
    if k%50 == 0:   # every 50 ms
        vs = 1 - vs

    # storage
    x_t[:,k] = x
    i_t[:,k] = [iR, iC]
    r_t[k] = Rp

    # derivatives 
    Dx = np.array([(vs - x[1])/L, (x[0] - iR)/C])

    # integrals
    x += Dx*T

# Save to file
store(x_t, "data/states")

# Plot
plt.figure()
plt.plot(t*1000, x_t.T)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('State values')
plt.legend(('Current (A)', 'Voltage (V)'))

plt.figure()
plt.plot(t*1000, i_t.T)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('Current (A)')
plt.legend(('Resistor Current (A)', 'Capacitor Current (A)'))

plt.figure()
plt.plot(t*1000, r_t)
plt.xlabel('Time, $t$ (ms)')
plt.ylabel('Impedance')
plt.legend(('Impedance (Ohm)'))

plt.show()


