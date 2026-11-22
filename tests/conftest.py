"""Shared fixtures: small synthetic problems so the suite runs in seconds without any data or LLM."""
from __future__ import annotations

import os

import numpy as np
import pytest

from sim.simulators import synthetic as synth
from sim.testbeds.synthetic import make_synthetic_problem
from sim.testbeds.toy import toy_problem, toy_simulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AASB_PARQUET = os.path.join(ROOT, "data", "raw", "AASB_Climate_Dataset", "data")
ORG_GROUPS = os.path.join(ROOT, "data", "org_groups.csv")
HAS_AASB = os.path.isdir(AASB_PARQUET) and os.path.exists(ORG_GROUPS)

requires_aasb = pytest.mark.skipif(not HAS_AASB, reason="private AASB dataset not downloaded (see README)")


@pytest.fixture(scope="session")
def root_dir():
    return ROOT


@pytest.fixture()
def rng():
    return np.random.default_rng(12345)


@pytest.fixture(scope="session")
def small_problem():
    """4 groups x (10, 8, 7, 6) organisations x 6 decisions, ~75% response."""
    return make_synthetic_problem(group_sizes=(10, 8, 7, 6), n_dec=6, response_rate=0.75, seed=3)


@pytest.fixture(scope="session")
def tiny_problem():
    """3 groups x (6, 5, 4) organisations x 4 decisions: small enough to enumerate every pilot."""
    return make_synthetic_problem(group_sizes=(6, 5, 4), n_dec=4, response_rate=0.8, seed=11)


@pytest.fixture(scope="session")
def noisy_sim(small_problem):
    return synth.iid_noise(small_problem, 0.3, np.random.default_rng(7))


@pytest.fixture(scope="session")
def structured_sim(small_problem):
    return synth.structured_misreading(small_problem, group=1, e_s=0.9, rng=np.random.default_rng(8), k=2,
                                       n_members=4)


@pytest.fixture(scope="session")
def toy():
    prob = toy_problem(n_per_group=4)
    return prob, toy_simulator(prob)
