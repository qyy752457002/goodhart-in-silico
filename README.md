# Goodhart in Silico

**Rectify-then-certify policy design with LLM-simulated stakeholders.**

Regulators increasingly pre-test draft proposals on LLM "synthetic respondents". When a draft is
*optimised* against such a simulator, the optimiser picks the options the simulator is most
optimistic about, so the support it actually gets falls short of the support predicted. Under
egalitarian objectives, the group the simulator misreads becomes the real loser while the design
still looks balanced in simulation (*illusory fairness*).

This repository contains:

- **a measurement protocol** for this effect: Goodhart curves under rising optimisation pressure
  and their decomposition by group;
- **GUARD**, which uses a small stratified pilot of real responses twice. First it *rectifies* the
  simulator per stakeholder group and proposal before optimising. Then it *certifies* the selected
  package with a lower bound on true welfare that holds for all packages at once;
- **all baselines and ablations** it is compared with;
- **testbeds**: organisations' stances on Australia's climate-disclosure exposure draft (AASB ED
  SR1), an adapter for OpinionQA survey data, and synthetic data.

## Repository layout

```
goodhart-in-silico/
├── code/sim/                 Python package (import name: sim)
│   ├── data.py               stances, support mappings, grouping schemes
│   ├── welfare.py            affine utilities, group welfare, power means
│   ├── optimize.py           package enumeration, max-min MILP, best-of-N law
│   ├── bounds.py             finite-population confidence bounds
│   ├── certify.py            mis-set mass, uniform bounds on group welfare
│   ├── pilot.py              stratified pilot designs
│   ├── engine.py             precomputation and evaluation for one (problem, simulator) pair
│   ├── guard.py              GUARD: rectify, certify, variants
│   ├── baselines.py          pilot-only, plug-ins, ensemble pessimism, HedgeTune
│   ├── methods.py            registry of decision rules and certificates
│   ├── goodhart.py           Goodhart curves, NDH, functional-form fit
│   ├── metrics.py, stats.py  validity metrics; bootstrap, Wilcoxon + Holm, Clopper-Pearson
│   ├── simulators/           LLM personas, prompts, parsing, backends, cache, fitted models
│   ├── testbeds/             aasb, opinionqa, synthetic, toy
│   ├── experiments/          describe, simulate, rq1, rq2, rq3, synthetic, sensitivity, toy
│   ├── reporting/            paper tables and figures
│   └── cli.py                command-line entry point (python -m sim)
├── configs/                  default (scenario S1), s2, smoke test, OpinionQA template
├── data/
│   ├── org_groups.csv        organisation → stakeholder type and group
│   ├── board_decisions.csv   the Board's final decision per proposal (diagnostic only)
│   ├── org_profiles.csv      optional anonymous persona descriptions
│   └── simulated/aasb/       precomputed LLM stance distributions (GPT and Llama)
├── results/
│   ├── aasb/paper/           tables and figures, scenario S1
│   └── s2/aasb/paper/        tables and figures, scenario S2
├── tests/                    unit and end-to-end tests (no LLM needed)
└── docs/IMPLEMENTATION_NOTES.md
```

## Quick start

Requires Python ≥ 3.10.

```bash
pip install -e ".[all]"
python -m pytest                 # no data and no LLM needed
python -m sim fetch-data         # download the AASB dataset into data/raw/ (access required, see below)
python -m sim all                # run all experiments with the committed simulators (CPU only)
```

`data/simulated/` holds the LLM outputs, so once you have the dataset, every experiment can be
reproduced **without calling a model**.

## Data availability

> **The AASB stance dataset is private and is not included in this repository.**

The experiments on the AASB testbed use staff-annotated stances of organisations on AASB ED SR1.
These are hosted as a **private** Hugging Face dataset (`testbed.aasb.hf_repo` in
`configs/default.yaml`). To get access:

1. Request access from the authors (open an issue in this repository).
2. Once access is granted, put a Hugging Face token in `.env` (`HF_TOKEN=...`, see `.env.example`).
3. Run `python -m sim fetch-data`.

Without the dataset you can still:

- run the test suite (the 5 tests that need the dataset are skipped);
- run the full pipeline on synthetic data (`configs/smoke.yaml`) and the toy example (`python -m sim toy`);
- inspect the published tables and figures in `results/`.

This repository contains only material derived from the dataset: the LLM-predicted stance
distributions in `data/simulated/`, the organisation-to-group mapping in `data/org_groups.csv`, and
aggregate results in `results/`. It does not contain the annotated stances themselves. Please do
not redistribute the dataset.

## Pipeline

All commands read `configs/default.yaml`. You can add your own config file with `--config` and
override single values with `--set key.sub=value`. Outputs go to `results/<testbed>/<experiment>/`.

| Command | Output |
|---|---|
| `python -m sim describe` | testbed statistics |
| `python -m sim rq1` | Table 1: simulator validity, bias matrices, leakage contrasts |
| `python -m sim rq2` | Figs. 2–3: Goodhart curves, group decomposition |
| `python -m sim rq3` | Table 2, Fig. 4: GUARD vs baselines, coverage, pilot-size curve |
| `python -m sim synthetic` | Fig. 5: structured misreading, coverage |
| `python -m sim sensitivity` | sensitivity analyses |
| `python -m sim report` | all tables (.csv/.tex) and figures (.pdf/.png) → `results/<testbed>/paper/` |
| `python -m sim all` | all of the above; add `--set project.n_jobs=8` to run in parallel |

Scenario S2 (context personas and relational models):

```bash
python -m sim all -c configs/s2.yaml --steps rq1 rq2 rq3 report
```

### Re-running the LLM simulators (optional)

```bash
cp .env.example .env                              # then fill in OPENAI_API_KEY
python -m sim simulate -m r1a_gpt --dry-run       # print prompts and call counts; nothing is sent
python -m sim simulate -m r1a_gpt -m r1b_gpt -m r1a_llama -m r1b_llama
python -m sim status
```

| Backend | Use |
|---|---|
| `openai_compat` | any OpenAI-compatible endpoint; the key is read from the variable named in `api_key_env` |
| `ollama` | local models, e.g. `llama3.1:8b-instruct-q4_K_M`; sets `num_ctx` so long prompts are not truncated |
| `mock` | tests and dry runs only |

Every reply is cached in `data/cache/`, keyed by a hash of the request, so an interrupted run
resumes where it stopped.

To check the wiring end to end on synthetic data with a mock LLM:

```bash
python -m sim simulate -c configs/smoke.yaml -m mock_named -m mock_anon
python -m sim all -c configs/smoke.yaml
```

## More

- [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md): implementation choices, running
  time and what the tests check.
- The AASB stance dataset is private; see [Data availability](#data-availability).
