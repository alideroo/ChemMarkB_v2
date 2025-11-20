# ChemMarkB

## Install packages

You can install the requirements via conda.
```
conda env create -f environment.yaml
```

## Download and install this package

```
git clone https://github.com/alideroo/ChemMarkB_v1
cd # ChemMarkB
conda activate chemmarkb
pip install -e .
```

## Projecting molecules into the synthesizable space

Training your own # ChemMarkB

```
cd scripts
python train.py ./configs/synthesis_net.yml
```
Example:

```
cd scripts
python sample.py \
    --checkpoint $CPATH \
    --input $IPATH \
    --output $OPATH
```
The results are saved in $OPATH.
## Contact
rczpku@163.com.
