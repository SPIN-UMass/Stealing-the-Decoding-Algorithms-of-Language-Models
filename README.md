# Stealing-the-Decoding-Algorithms-of-Language-Models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Our algorithm offers multiple arguments to run experiments for each stage. Below is an example of how to execute the code:

```bash
python Main.py --stage 3 --targeted_model 'gpt2' --temperature 0.8 --algorithm 'temperature'
```

Furthermore, an example of how to assess a particular estimation using the metrics proposed in the paper can be found below:

```bash
python Evaluation.py --targeted_model 'gpt2' --original_temperature 0.8 --estimated_temperature 0.801
```
