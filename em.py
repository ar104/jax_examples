import argparse
import jax
from jax import numpy as jnp
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import norm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--jax', action='store_true')
parser.add_argument('--examples', type=int, default=1000000)
args = parser.parse_args()


def create_gm_dataset(n_examples: int, p=0.8, mu0=-1, sd0=0.1, mu1=1,
                      sd1=0.1) -> np.array:
    '''Gaussian mixture dataset. Draw from one of two normal distributions
       with choice being dictated by a bernoulli distribution.
    '''
    latents = np.random.binomial(n=1, p=p, size=n_examples)
    choice1 = np.random.normal(loc=mu0, scale=sd0, size=n_examples)
    choice2 = np.random.normal(loc=mu1, scale=sd1, size=n_examples)
    return np.choose(latents, (choice1, choice2))


def em(examples, max_iters, tol):
    ''' Expectation maximization implementation.'''
    mu0 = -2.0
    sd0 = 1.0
    mu1 = +2.0
    sd1 = 1.0
    p = 0.5
    start = time.time()
    for iter in range(1, 1 + max_iters):
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
    stop = time.time()
    print(f'iters = {iter} EM sec/iter = {(stop - start)/iter}')
    return p, mu0, sd0, mu1, sd1


def jax_em_iter(examples, p, mu_0, sd_0, mu_1, sd_1):
    p_x_z_is_0 = jax.scipy.stats.norm.pdf(examples, loc=mu0, scale=sd0)
    p_x_z_is_1 = jax.scipy.stats.norm.pdf(examples, loc=mu1, scale=sd1)
    p_x = p_x_z_is_0*(1 - p) + p_x_z_is_1*p
    # Setup latent distribution using current estimate.
    p_z_given_x = p_x_z_is_1*p/p_x
    p_new = jnp.mean(p_z_given_x)
    p_not_z_given_x = 1.0 - p_z_given_x
    # Maximize expectation.
    sum_p_given_x = jnp.sum(p_z_given_x)
    sum_not_p_given_x = jnp.sum(p_not_z_given_x)
    mu1_new = jnp.dot(p_z_given_x, examples)/sum_p_given_x
    mu0_new = jnp.dot(p_not_z_given_x, examples)/sum_not_p_given_x
    sd1_new = jnp.sqrt(
        jnp.dot(p_z_given_x, (examples - mu1_new)**2)/sum_p_given_x)
    sd0_new = jnp.sqrt(
        jnp.dot(p_not_z_given_x, (examples - mu0_new) ** 2) /
        sum_not_p_given_x)
    return p_new, mu0_new, sd0_new, mu1_new, sd1_new


def jax_em(examples, max_iters, tol):
    ''' Expectation maximization implementation using JAX.'''
    examples = jax.device_put(examples)
    mu0 = -5.0
    sd0 = 2.0
    mu1 = +5.0
    sd1 = 2.0
    p = 0.01
    jited_core_loop = jax.jit(jax_em_iter)
    start = time.time()
    for iter in range(1, 1 + max_iters):
        p_new, mu0_new, sd0_new, mu1_new, sd1_new = jited_core_loop(
            examples, p, mu0, sd0, mu1, sd1)
        p_new, mu0_new, sd0_new, mu1_new, sd1_new = (
            p_new.item(),
            mu0_new.item(),
            sd0_new.item(),
            mu1_new.item(),
            sd1_new.item())
        max_delta = max(
            abs(p_new - p),
            abs(mu0_new - mu0),
            abs(mu1_new - mu1),
            abs(sd0_new - sd0),
            abs(sd1_new - sd1))
        if max_delta < tol:
            break
        p, mu0, sd0, mu1, sd1 = p_new, mu0_new, sd0_new, mu1_new, sd1_new
    stop = time.time()
    print(f'JAX iters = {iter} EM sec/iter = {(stop - start)/iter}')
    return p, mu0, sd0, mu1, sd1


mu0, sd0, mu1, sd1, p = -1.0, 0.1, 1.0, 0.1, 0.8
print(f'Model parameters = p={p}, mu0={mu0}, sd0={sd0}, mu1={mu1}, sd1={sd1}')
print(f'Creating dataset with {args.examples} examples.')
dataset = create_gm_dataset(n_examples=args.examples)
if args.jax:
    p_fit, mu0_fit, sd0_fit, mu1_fit, sd1_fit = jax_em(dataset, 10, 0.001)
else:
    p_fit, mu0_fit, sd0_fit, mu1_fit, sd1_fit = em(dataset, 10, 0.001)
print(f'Fit parameters = p={p_fit}, mu0={mu0_fit}, sd0={sd0_fit}, '
      f'mu1={mu1_fit}, sd1={sd1_fit}')
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].hist(dataset, bins=200)
axes[0].set_title('Observed Data Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count')
generated = create_gm_dataset(
    n_examples=args.examples, p=p_fit, mu0=mu0_fit, sd0=sd0_fit, mu1=mu1_fit,
    sd1=sd1_fit)
axes[1].hist(generated, bins=200)
axes[1].set_title('Generated Data Distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Count')
# plt.show()
fig.savefig('em_data_distribution.png')
