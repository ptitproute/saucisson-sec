#! /bin/bash

categorie() {
    # couleur aléatoire parmi la liste
    # bleue -> 34, violet -> 35, cyan -> 36, blanc -> 37
    couleur=${couleur:=32}
    uppercase=$(echo -E "$1" | awk '{ print toupper($0) }')
    echo
    echo "################################################################"
    echo -e "          \e[1;${couleur}m${uppercase}\e[0m"
    echo "################################################################"
    echo
}
