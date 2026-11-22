"""SIM -- Goodhart in Silico: rectify-then-certify policy design with LLM-simulated stakeholders.

Layers (each depends only on those above it):

  data, welfare, optimize        the decision problem: stances -> affine utilities -> packages
  bounds, certify, pilot         finite-population confidence bounds, uniform welfare bounds
  simulators/                    stance distributions: LLM personas, fitted models, synthetic
  engine, guard, baselines,      decision rules and certificates for one (problem, simulator) pair
  methods
  goodhart, metrics, stats       measurement protocols (RQ2, RQ1) and statistical reporting
  testbeds/                      AASB ED SR1, OpinionQA, synthetic, toy
  experiments/, reporting/, cli  the pipeline that produces the paper's tables and figures
"""

__version__ = "0.2.0"
