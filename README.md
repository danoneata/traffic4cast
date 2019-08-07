# Traffic4cast

predicting traffic for the NeurIPS'19 challenge

# Setup

1. Install anaconda or miniconda.
2. Install dependencies and activate environment:
```bash
conda env create environment.yml
conda activate traffic4cast
```

# Distributed hyper-parameter tuning

Start the server, for example on `lenovo-storage`:

```bash
python hypertune.py --run-id test --n-workers 4
```

Start the workers, for example on other lenovo's – `lenovo{3,4,6,7}`:

```bash
python hypertune.py --run-id test --worker
```

The hyper-parameter tuning can be runned also locally, but it's unlikely to be efficient:

```bash
python hypertune.py --n_workers 4 &
for i in {1..4}; do
	sleep 0.1
	python hypertune.py --worker &
done
```
