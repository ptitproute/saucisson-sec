#! /bin/bash 

source lib/chargeur.sh
source lib/affichage/code_couleurs

dossier=affichage chargeur message categorie 

categorie "début de l'installation" 35

# Installe la liste de paquets prérequis
apt install $(cat requis/paquets) -y && couleur=reussite message "paquets prérequis installés" || couleur=erreur message "erreur lors de l'installation des paquets prérequis"

# Installe la liste de paquets python via un environnement virtuel
pyenv activate venv && couleur=reussite message "environnement virtuel installés" || couleur=erreur message "erreur lors de la création de l'environnement virtuel"

# Installe les paquets python requis dans l'environnement virtuel
pip install -r requis/python && couleur=$reussite message "paquets python installés" || couleur=$erreur message "erreur lors de l'installation des paquets python"
