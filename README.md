# Echo-Bio

Evidence-Constrained Harness Optimization for Efficient Biomedical Research Agents.

Echo-Bio is a user-facing research runtime around a patched Biomni agent. It keeps executed code,
raw observations, artifacts, scientific receipts and claim–evidence links in append-only local
stores while using a strict provider-native action interface. The first public preview supports
one model profile only: DeepSeek V4.1 Flash (`deepseek-flash`) through the official Responses API.

This repository is intentionally an **application release**, not the internal experiment archive.
It contains no benchmark corpora, frozen paid responses, private rubrics, campaign authorizations,
expert labels or experiment-reproduction scripts.

## What a user can do

- run a biomedical question with optional local data files;
- allow Biomni to execute Python, R or shell actions in a bounded loop;
- retain a content-chained trace, content-addressed artifacts and evidence receipts;
- inspect the exact answer and durable run outputs after completion;
- fail closed on an unknown provider response, with no automatic retry or transport fallback.

Echo-Bio does **not** certify biological truth. Its records establish execution and provenance
boundaries; scientific conclusions still require domain review and appropriate validation.

## Install

Linux, Python 3.12 and R 4.4.3 are the recommended profile. Python 3.11+ is supported by the
package. Create the environment:

```bash
mamba create -n echo-bio -c conda-forge python=3.12 r-base=4.4.3
mamba activate echo-bio
python -m pip install --upgrade pip
```

Echo-Bio currently requires one small hook patch on Biomni commit
`400c1f366b96a35ca253e13c9b06c5076af41d65`:

```bash
git clone https://github.com/Zimiao1025/Echo-Bio.git
cd Echo-Bio

git clone https://github.com/snap-stanford/Biomni.git vendor/Biomni
git -C vendor/Biomni checkout 400c1f366b96a35ca253e13c9b06c5076af41d65
git -C vendor/Biomni apply ../../patches/biomni-a1-echo-hooks.patch

python -m pip install -e vendor/Biomni
python -m pip install -e ".[biomni]"
echo-bio doctor --require-runtime
```

The patch adds identity-preserving extension hooks; it does not change unadapted Biomni behavior.

## Configure DeepSeek

Keys are read only from the process environment. Do not put a key in this repository.

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
echo

# Zero-cost profile inspection
echo-bio provider-check

# One explicitly authorized paid ABI canary
echo-bio provider-check --live --confirm-paid
```

## Run your own task

The scientific executor runs local model-generated code. Use a disposable environment or sandbox,
inspect your data permissions and pass both confirmations deliberately:

```bash
echo-bio run \
  --task "Compare treatment and control expression, report effect sizes and limitations." \
  --data "$PWD/data/expression.csv=normalized gene-expression matrix" \
  --workspace "$PWD/.echo-bio/runs" \
  --max-requests 8 \
  --confirm-paid \
  --allow-local-execution
```

For a longer prompt, use `--task-file question.md`. Each run creates its own directory containing
`answer.md`, `summary.json`, the hash-chained trace, evidence ledger and content-addressed artifacts.
Echo-Bio does not retry a lost response automatically, so a failed-closed run may still have incurred
provider cost; inspect your provider account before starting another run.

See [Getting started](docs/getting-started.md), [runtime and safety](docs/runtime-safety.md), and the
[DeepSeek provider profile](docs/deepseek-v41-flash.md) for the complete boundaries.

## Development checks

No live call is made by these commands:

```bash
python -m pip install -e ".[biomni,dev]"
pytest -q
ruff check src tests
```

## Release status and citation

Version `0.1.0rc1` is a public preview. A software license still needs to be selected before other
people have permission to redistribute or modify the code.

Hongliang Li. “Echo-Bio: When Cost Minimization Fails for Biomedical Research Agents.” 2026.
https://zimiao1025.github.io/blog/2026/10/27/echo-bio/
