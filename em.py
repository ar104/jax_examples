import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm


_NORM_SD = 0.1


def create_gm_dataset(n_examples: int, p=0.8, mu0=-1, mu1=1) -> np.array:
    '''Gaussian mixture dataset. Draw from one of two normal distributions
       with choice being dictated by a bernoulli distribution.
    '''
    latents = np.random.binomial(n=1, p=p, size=n_examples)
    choice1 = np.random.normal(loc=mu0, scale=_NORM_SD, size=n_examples)
    choice2 = np.random.normal(loc=mu1, scale=_NORM_SD, size=n_examples)
    return np.choose(latents, (choice1, choice2))


def em(examples, max_iters, tol):
    ''' Expectation maximization implementation.'''
    mu0 = -2.0
    sd0 = 1.0
    mu1 = +2.0
    sd1 = 1.0
    p = 0.5
    for iter in range(max_iters):
        p_x_z_is_0 = norm.pdf(examples, loc=mu0, scale=sd0)
        p_x_z_is_1 = norm.pdf(examples, loc=mu1, scale=sd1)
        p_x = p_x_z_is_0*(1 - p) + p_x_z_is_1*p
        # Setup latent distribution using current estimate.
        p_z_given_x = p_x_z_is_1*p/p_x
        p_new = np.mean(p_z_given_x).item()
        p_not_z_given_x = 1.0 - p_z_given_x
        # Maximize expectation.
        sum_p_given_x = np.sum(p_z_given_x).item()
        sum_not_p_given_x = np.sum(p_not_z_given_x).item()
        mu1_new = np.dot(p_z_given_x, examples).item()/sum_p_given_x
        mu0_new = np.dot(p_not_z_given_x, examples).item()/sum_not_p_given_x
        sd1_new = math.sqrt(
            np.dot(p_z_given_x, (examples - mu1_new)**2).item()/sum_p_given_x)
        sd0_new = math.sqrt(
            np.dot(p_not_z_given_x, (examples - mu0_new) ** 2).item() /
            sum_not_p_given_x)
        max_delta = max(
            abs(p_new - p),
            abs(mu0_new - mu0),
            abs(mu1_new - mu1),
            abs(sd0_new - sd0),
            abs(sd1_new - sd1))
        if max_delta < tol:
            break
        p, mu0, sd0, mu1, sd1 = p_new, mu0_new, sd0_new, mu1_new, sd1_new
        # print(f'{iter}: p={p}, mu0={mu0}, sd0={sd0}, mu1={mu1}, sd1={sd1}')
    return p, mu0, sd0, mu1, sd1


dataset = create_gm_dataset(n_examples=100000)
print(em(dataset, 10, 0.001))
fig, ax = plt.subplots()
ax.hist(dataset, bins=200)
ax.set_title('Observed Data Distribution')
ax.set_xlabel('Value')
ax.set_ylabel('Count')
# plt.show()
# fig.savefig('em_data_distribution.png')
