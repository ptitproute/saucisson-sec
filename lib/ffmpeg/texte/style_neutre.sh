#!/bin/bash#!/bin/bash

source lib/affichage/message.sh

style_neutre(){


    ## Utilise ffmpeg pour ajouter un texte centrée au mileu de la vidéo
    # $1: fichier d'entrée
    # $2: texte à afficher

    nom_du_fichier="data/videos/shader/$1"
    echo DEBUG: nom_du_fichier $nom_du_fichier
    texte="$2"
    echo DEBUG: texte $texte
    fichier_de_sortie="data/videos/final/$1"
    echo DEBUG: fichier_de_sortie $fichier_de_sortie
    echo 

    duree_videos_en_seconde=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $nom_du_fichier)
    echo DEBUG: duree_videos_en_seconde $duree_videos_en_seconde
    debut=$(echo "$duree_videos_en_seconde * 0.20" | bc)
    echo DEBUG: debut $debut
    fin=$(echo "$duree_videos_en_seconde * 0.80" | bc)
    echo DEBUG: fin $fin

    # --- 


    ## Charge les variables d'environnement liés 

    # Couleur du texte 
    font_color=${font_color:-$FONT_COLOR}

    # Police du texte - default Roboto-regular.ttf
    font_file=${font_file:-$FONT_FILE}

    # Taille de police - default 24
    font_size=${font_size:-$FONT_SIZE}

    # --- 

    ##

    ffmpeg -i "$nom_du_fichier" \
    -vf drawtext="fontfile=lib/police/$font_file:text='${texte}':fontcolor=$font_color:fontsize=$font_size:x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,$debut,$fin)'" \
    -codec:a copy $fichier_de_sortie && message "Application du style neutre réussie" || message "Application du style neutre échouée"

    # ffmpeg -i "$nom_du_fichier" -vf drawtext="fontfile=lib/police/Roboto-regular.ttf: \
    # text='$texte': fontcolor=white: fontsize=24: box=1: boxcolor=black@0.5: \
    # boxborderw=5: x=(w-text_w)/2: y=(h-text_h)/2" $fichier_de_sortie && message "Application du style neutre réussie" || message "Application du style neutre échouée"

}