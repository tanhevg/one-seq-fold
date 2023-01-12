# Fold classification of CASP targets based on ECOD

```
$ python casp_eval.py --templates templates.fasta --query casp.fasta --model_weights checkpoint.pt --device cuda:0 --out_file folds.csv
$ python casp_eval.py -h
usage: Evaluation on CASP Targets [-h] --templates TEMPLATES --query QUERY
                                  --model_weights MODEL_WEIGHTS
                                  [--device DEVICE]
                                  [--minibatch_size MINIBATCH_SIZE]
                                  [--log_file LOG_FILE]
                                  [--log_level LOG_LEVEL]
                                  [--out_file OUT_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --templates TEMPLATES
                        Path to fasta file with template sequences
  --query QUERY         Path to fasta file with query sequence(s)
  --model_weights MODEL_WEIGHTS
                        Path to file with model checkpoint
  --device DEVICE       CUDA device, if available
  --minibatch_size MINIBATCH_SIZE
                        Minibatch size, depends on available memory
  --log_file LOG_FILE
  --log_level LOG_LEVEL
  --out_file OUT_FILE   Path for CSV output; stdout if not specified
```
