#! /bin/bash

message(){
    couleur=${couleur:=32}

    echo
    echo -e "\e[1;${couleur}m${1}\e[0m"
    echo
}