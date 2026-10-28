# Getting started

## 1. Install the supported runtime

Echo-Bio 0.1.0rc1 is qualified for Linux, Python 3.12 and R 4.4.3. R is needed only when an agent
chooses an R action.

```bash
mamba create -n echo-bio -c conda-forge python=3.12 r-base=4.4.3
mamba activate echo-bio
python -m pip install --upgrade pip
```

Clone Echo-Bio and the frozen Biomni revision:

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

`echo-bio doctor` is read-only. A missing key is informational. `--require-runtime` verifies that
patched Biomni is present; add `--require-r` if the planned task must execute R.

## 2. Configure the provider

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
echo
echo-bio provider-check
```

Run the optional one-request paid canary only after checking current provider pricing:

```bash
echo-bio provider-check --live --confirm-paid
```

## 3. Submit a research task

```bash
echo-bio run \
  --task-file question.md \
  --data "$PWD/data/counts.csv=sample-by-gene count matrix" \
  --workspace "$PWD/.echo-bio/runs" \
  --run-id my-first-run \
  --max-requests 8 \
  --confirm-paid \
  --allow-local-execution
```

`--data` is repeatable and must use `PATH=DESCRIPTION`. Echo-Bio hashes each input file into the run
manifest and exposes its resolved path and description to Biomni. It never copies the API key into a
run artifact.

## 4. Inspect the result

```text
.echo-bio/runs/my-first-run/
├── answer.md
├── summary.json
├── trace/my-first-run/run.json
├── trace/my-first-run/events.jsonl
├── evidence/my-first-run/evidence.jsonl
└── artifacts/
```

The answer is model-generated. The evidence ledger may record successful execution and provenance,
but it is not a substitute for biological interpretation, independent replication or human review.
