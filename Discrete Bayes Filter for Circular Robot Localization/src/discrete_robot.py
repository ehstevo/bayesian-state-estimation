import numpy as np
import matplotlib.pyplot as plt
import util

# constants
K = 100                 # time steps
pc = 0.8                # probability of reading correct color
pm = [0.2, 0.3, 0.5]    # pmf of moving backward, staying still, or moving forward

# estimation
def estimate(K, pc, pm, z_t, x_t, map):
    px = np.ones(len(map)) * (1/len(map))   # unknown starting point, initial pmf of position is uniform
    pz_x = util.get_pz_x(pc, map)           # probability of measurement given the state
    M = util.markov_transition(pm)          # markov transition matrix
    px_t = np.zeros((len(map), K))          # vector of px for each time step, each column represents a time step
    xh_t = np.zeros(K)                      # vector of position estimates
    accuracy = 0                            # running sum of accurate measurements
    validity = 0                            # validity of our estimator, ideally close to 1 
    goodness = 0                            # tells how often the best guess was sufficiently confident                         

    for i in range(K):
        #get measurement
        measurement = z_t[i]
        
        # update
        posterior = pz_x[measurement] * px
        posterior /= np.sum(posterior)

        # estimate position
        xh = np.argmax(posterior)

        # true position
        x = int(x_t[i])

        # determine if accurate, and calculate validity
        if (xh == x):
            accuracy += 1
            validity += (1/posterior[x]) * (1/K)

        # determine goodness of estimate
        if posterior[xh] > 0.5:
            goodness += 1

        # store
        px_t[:,i] = posterior
        xh_t[i] = xh

        # propagate: gives prior for next time step
        px = M @ px_t[:,i]

    return px_t, xh_t, validity, (goodness/K), (accuracy/K)

# does one monte carlo run of 100 simulations
# outputs validity, goodness, and accuracy
# plots the last run of the simulation
def monte_carlo():
    validity = 0
    goodness = 0
    accuracy = 0
    for i in range(100):
        # generate truth and measurements
        x_t, z_t, map = util.gen_truth(K, pc, pm)
        px_t, xh_t, v, g, a = estimate(K, pc, pm, z_t, x_t, map)
        validity += v
        goodness += g
        accuracy += a
        if i==99:
            util.plot_results(px_t, xh_t, x_t)
    validity /= 100
    goodness /= 100
    accuracy /= 100
    print(validity, goodness, accuracy)
    plt.show()

# does len(pc) * len(pm) monte carlo runs of 100 simulations
# one for each sensor accuracy and dynamics PMF value
# outputs an accuracy matrix, with the value of the estimator
# accuracy for each (sensor accuracy, dynamics PMF) pair
def monte_carlo_sets(pc, pm):
    accuracy_matrix = np.zeros((4, 8))
    for i in range(len(pm)):
        for j in range(len(pc)):
            accuracy = 0
            for _ in range(100):
                x_t, z_t, map = util.gen_truth(K, pc[j], pm[i])
                px_t, xh_t, v, g, a = estimate(K, pc[j], pm[i], z_t, x_t, map)
                accuracy += a
            accuracy_matrix[i,j] = accuracy/100
    return accuracy_matrix

monte_carlo()

# pc = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# pm = [[0.3, 0.3, 0.4], [0.2, 0.2, 0.6], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]
# am = monte_carlo_sets(pc, pm)
# plt.figure()
# plt.plot(am[0], label=f'PMF: {pm[0]}')
# plt.plot(am[1], label=f'PMF: {pm[1]}')
# plt.plot(am[2], label=f'PMF: {pm[2]}')
# plt.plot(am[3], label=f'PMF: {pm[3]}')
# locs = plt.xticks()[0][1:-1]
# plt.xticks(ticks=locs, labels=pc)
# plt.xlabel('Sensor Accuracy')
# plt.ylabel('Estimator Accuracy')
# plt.legend()
# plt.show()