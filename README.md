# SpecCourse

This project studies speculative decoding, focusing on finding optimal quantization for draft models to improve end-to-end latency.

## Setup

1. **Clone and Update Submodules**
```bash
git clone https://github.com/yourusername/SpecCourse.git
cd SpecCourse
git submodule update --init --recursive
```

2. **Create Virtual Environment** (using uv)
```bash
uv venv --python 3.12 --python-preference only-managed
source .venv/bin/activate
uv pip install -e .
```

3. **Install Dependencies**
```bash
python setup.py install
```

## Usage

To use the scripts go to `spec_course` directory:

```
cd spec_course
```

### 1. Model Quantization
Configure models and quantization schemes in `configs/quantization.yaml`, then run:
```bash
python scripts/quantize.py --config configs/quantization.yaml
```

### 2. Accuracy Evaluation
Set models and evaluation parameters in `configs/evaluate_lm_eval.yaml`, then run:
```bash
python scripts/evaluate_accuracy.py --config configs/evaluate_lm_eval.yaml
```

### 3. Speculative Decoding Experiments
Configure target and draft model setups in `configs/sd_setups.yaml`, then run:
```bash
python scripts/run_sd.py \
  --config configs/sd_setups.yaml \
  --dataset chat \
  --num_prompts 30 \
  --setup_type single_setup \
  --output_dir results/sd_experiments
```

### 4. Experiments with different RPS
Configure target and draft model setups in `configs/load_test.yaml`, then run:
```bash
python scripts/run_load_test.py --config configs/load_test.yaml
```

### 5. Data Analysis

Results are stored in the `results` directory. To analyze metrics using the database:

1. Import LM Evaluation metrics:
```bash
python database/run.py \
  --etl_class accuracy \
  --data_dir results/gsm8k/ \
  --db_name database.db
```

2. Import Speculative Decoding metrics:
```bash
python database/run.py \
  --etl_class sd_metrics \
  --data_dir results/sd_experiments/ \
  --db_name database.db
```

3. Import load test metrics:
```bash
python database/run.py \
  --etl_class load_test_metrics \
  --data_dir results/load_test \
  --db_name database.db
```

4. To view the analysis results, go to `notebook.ipynb`.

## Project Structure

```
deps/
spec_course/
├──.logs/
├── configs/
├── database/
├── models/
├── results/
└── scripts/
```
