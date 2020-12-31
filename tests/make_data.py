import numpy as np
import scipy.stats as st


def make_data(d, normality=1.0, asymmetry=0.5, difficulty=0.2,
              population=[100] * 10, normality_prec=5, difficulty_prec=5):
    difficulty_prec += d
    m = len(population)

    mu = st.multivariate_normal(np.zeros(d), np.eye(d)).rvs(size=m)
    cov = difficulty ** 2 * st.wishart(difficulty_prec, np.eye(d) / difficulty_prec).rvs(size=m)
    beta = asymmetry * np.array([st.multivariate_normal(np.zeros(d), c).rvs() for c in cov])
    lmd = st.invgauss(1 / normality_prec, 0, normality * normality_prec).rvs(size=m)

    z = np.concatenate([np.full(n, i) for i, n in enumerate(population)])
    y = st.invgauss(1/lmd[z], 0, lmd[z]).rvs()
    x = np.array([st.multivariate_normal(m, s).rvs() for m, s in zip(mu[z]+y[:,None]*beta[z], y[:,None,None] * cov[z])])
    return x, z + 1
