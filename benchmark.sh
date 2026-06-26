#!/usr/bin/env bash

# Vérifier si deux arguments sont fournis
if [ $# -ne 2 ]; then
    echo "Usage: $0 <dossier_de_sortie> <chemin_video>"
    exit 1
fi

# Arguments
OUTPUT_DIR=$1
VIDEO_PATH=$2

# Vérifier si la vidéo existe
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Erreur: le fichier vidéo '$VIDEO_PATH' n'existe pas."
    exit 1
fi

# Créer le dossier de sortie s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# Fichiers de sortie
TIME_RESULTS="$OUTPUT_DIR/time.result"
rm -f $TIME_RESULTS

# Liste des tags
TAGS=$(git tag | grep -E '^(gpu|cpu)-[0-9]+$')
TOTAL_TAGS=$(echo "$TAGS" | wc -l)
CURRENT_TAG=0

for TAG in $TAGS; do
    ((CURRENT_TAG++))
    MODE=${TAG%%-*}  # Extraire 'gpu' ou 'cpu' du tag
    echo "Tag $CURRENT_TAG/$TOTAL_TAGS: checkout en cours"

    # Checkout sur le tag
    git checkout $TAG > /dev/null 2>&1 || { echo "Erreur: impossible de checkout sur $TAG"; exit 1; }

    echo "Tag $CURRENT_TAG/$TOTAL_TAGS: nettoyage du build"
    # Nettoyage et construction
    rm -rf build
    mkdir build

    echo "Tag $CURRENT_TAG/$TOTAL_TAGS: build en cours"
    cmake -S . -B build > /dev/null 2>&1 || { echo "Avertissement: cmake échoué pour $TAG"; exit 1; }
    make -C build -Bj > /dev/null 2>&1 || { echo "Avertissement: premier make échoué pour $TAG";}
    make -C build -j > /dev/null 2>&1 || { echo "Erreur: second make échoué pour $TAG"; continue;}

    echo "TIME $TAG" >> $TIME_RESULTS
    if [ "$MODE" == "gpu" ]; then
        echo "Tag $CURRENT_TAG/$TOTAL_TAGS: time et profiling en cours"
        { time nvprof ./build/stream --mode="$MODE" "$VIDEO_PATH" --output=output_"$TAG".mp4 &> "$OUTPUT_DIR/profiling.$TAG"; } 2>> $TIME_RESULTS
    else
        echo "Tag $CURRENT_TAG/$TOTAL_TAGS: time en cours"
        { time ./build/stream --mode="$MODE" "$VIDEO_PATH" --output=output_"$TAG".mp4; } 2>> $TIME_RESULTS
    fi
done

# Rétablissement à la branche principale
git checkout main > /dev/null 2>&1
mv *.mp4 "$OUTPUT_DIR"/

echo "Benchmark terminé. Résultats dans $OUTPUT_DIR."
