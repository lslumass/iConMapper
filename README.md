# iConMapper: An Automated mapping tool for iConMetabolites   
This is a retrained DSGPM-TP model using [iConMetabolome](https://github.com/lslumass/iConMetabolome) database. For model details, please follow the original papers of [DSGPM](http://dx.doi.org/10.1039/D0SC02458A) and [DSGPM-TP](https://doi.org/10.1021/acs.jctc.4c01466)   

## Training
```

```

## Mapping (Prediction)
use **MappingPrediction.py** to predict the mapping of new metoblites   
```
usage: MappingPrediction.py [-h] [--name NAME] --smiles SMILES [--output OUTPUT] [--num NUM] [--labels] [--style STYLE]

iConMapper — CG mapping via DSGPM-TP

options:
  -h, --help       show this help message and exit
  --name NAME      Molecule name (default: molecular formula)
  --smiles SMILES  SMILES string of the molecule
  --output OUTPUT  Output directory (default: ./aa2cg)
  --num NUM        Number of CG beads (default: n_heavy_atoms // 3)
  --labels         Overlay bead type labels on the output image
  --style STYLE    Visualization style: 1=colored beads, 2=circles (default: 1)
```
