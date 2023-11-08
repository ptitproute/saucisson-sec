#!/bin/bash

chargeur(){
local fichiers=("$@")
for s in "${fichiers[@]}"; do
  source "${DOSSIER_COURANT}/lib/${dossier}/${s}.sh"
done
}