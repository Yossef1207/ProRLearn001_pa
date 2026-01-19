#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yossef.albuni@tuhh.de
#SBATCH --time 23:00:00
# --mem-per-cpu=2000
#SBATCH --gres gpu:1
#SBATCH --mem-per-gpu 75000
#SBATCH --output output/%x_%j.log

# Lokales Arbeits- und Eingabeverzeichnis definieren
WORKDIR=$SCRATCH/$SLURM_JOB_ID
INPUTDIR=$SLURM_SUBMIT_DIR

mkdir -p "$WORKDIR"

# Caches auf das lokale Verzeichnis legen (wichtig fuer HF, Pip, Triton)
export HF_HOME=$WORKDIR/hf_cache
export PIP_CACHE_DIR=$WORKDIR/pip_cache
export TRITON_HOME=$WORKDIR/triton

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TRITON_HOME"

# Benoetigte Daten/Code auf die lokale Platte kopieren
cp -r "$INPUTDIR" "$WORKDIR/"


# Load anaconda und aktiviere das prorlearn001-Env korrekt im Batch-Skript
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate prorlearn001-gpu

nproc_per_node=1

########################   ProRLearn001 Training   ########################

cd "$WORKDIR/ProRLearn001/ProRLearn" 

#python load_model.py  2>&1 | tee "load_model.log"

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

# Zielordner fuer Modelle im Submit-Verzeichnis anlegen
mkdir -p "$INPUTDIR/models"

for DATASET in "${DATASETS[@]}"; do
    echo ">>> Starte ProRLearn001 Klassifikation fuer Dataset: $DATASET"
    python VPG-classfication.py --dataset "${DATASET}" 2>&1 | tee "${DATASET}.log"
    echo ">>> ProRLearn001 Klassifikation fuer $DATASET abgeschlossen."

    # Log zurueckkopieren
    cp "$WORKDIR/ProRLearn001/ProRLearn/${DATASET}.log" "$INPUTDIR/logs/"

    # Modell nur kopieren, wenn es existiert (z.B. kein Abbruch im Training)
    MODEL_SRC="$WORKDIR/ProRLearn001/models/${DATASET}_best_model.pt"
    MODEL_DST="$INPUTDIR/models/${DATASET}_best_model.pt"
    if [[ -f "$MODEL_SRC" ]]; then
      cp "$MODEL_SRC" "$MODEL_DST"
    else
      echo "[WARN] Kein Model-Checkpoint gefunden fuer $DATASET ($MODEL_SRC)."
    fi
done

# Aufraeumen
rm -rf "$WORKDIR"
exit 0
