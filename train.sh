#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yossef.albuni@tuhh.de
#SBATCH --time 1-23:00:00
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 75000
#SBATCH --output output/%x_%j.log

# Lokales Arbeits- und Eingabeverzeichnis definieren
WORKDIR=$SCRATCH/$SLURM_JOB_ID
INPUTDIR=$SLURM_SUBMIT_DIR

mkdir -p "$WORKDIR"

# Caches auf das lokale Verzeichnis legen
export HF_HOME=$WORKDIR/hf_cache
export PIP_CACHE_DIR=$WORKDIR/pip_cache
export TRITON_HOME=$WORKDIR/triton

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_HOME"

# Benoetigte Daten/Code auf die lokale Platte kopieren
cp -r "$INPUTDIR"/* "$WORKDIR/"

# Load anaconda und aktiviere das prorlearn001-Env
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate prorlearn001-gpu

# # Zielordner fuer Modelle/Logs im Submit-Verzeichnis anlegen
# mkdir -p "$INPUTDIR/models"
# mkdir -p "$INPUTDIR/logs"
# mkdir -p "$INPUTDIR/output"

cd "$WORKDIR/ProRLearn" 

DATASETS=(
  "reposvul_dataset"
  "reposvul_with_codellama"
  "reposvul_with_gpt-4o"
  "reposvul_with_vul_codellama"
  "reposvul_with_vul_gpt-4o"
  "primevul_dataset"
  "primevul_with_codellama"
  "primevul_with_gpt-4o"
  "primevul_with_vul_codellama"
  "primevul_with_vul_gpt-4o"
)

for DATASET in "${DATASETS[@]}"; do
    echo ">>> Starte ProRLearn001 Klassifikation fuer Dataset: $DATASET"
    python VPG-classfication.py --dataset "${DATASET}" 2>&1 | tee "${DATASET}_512.log" 
    echo ">>> ProRLearn001 Klassifikation fuer $DATASET abgeschlossen."
    cp "$WORKDIR/ProRLearn/${DATASET}_512.log" "$INPUTDIR/logs_512/"
    MODEL_SRC="$WORKDIR/models_512/${DATASET}_best_model.pt"
    MODEL_DST="$INPUTDIR/models_512/${DATASET}_best_model.pt"
    cp "$MODEL_SRC" "$MODEL_DST"
done

echo ">>> Training abgeschlossen"
rm -rf "$WORKDIR"
exit 0