# TRAVAUX EN COURS 

![logo](images/logo.webp)

# Saucisson sec, une interface cli pour générer des transitions vidéos

Ce repo à pour but de regrouper différentes sources / programmes.
Simplifier la génération d'une page de transitions pour du format vidéos.

Ces transitions sont générés via les shaders openGl que l'on retrouve sur le site de shadertoy et une incrustation textuelle via ffmpeg :
https://www.shadertoy.com/

L'objectif est de fournir une syntaxe simple :

```
url:hauteurXlargeur:texte/cheminVersFichiers:style
```

## Apercu

Pour avoir un aperçu de ce qui est possible de faire

```
./main --demo
```

## Tester une url de shadertoy

Ou url est le chemin vers la page de shaders, vous pouvez tester l'url via :

```
./main --url "https://www.shadertoy.com/view/wtVyWK"
```

| Code de retours | Status          | message de sortie |
| :--------------- |:---------------:| -----:|
| Aligné à gauche  |   centré        |  Aligné à droite |

## Fonctionnement

Le programme vient récupérer le contenue de la page html, y exerce une extraction des balises <code>openGL</code> via l'expression régulière contenue dans la variable d'environnement $EXTRAIT_OPENGL_CODE.

Il récupère ensuite les dimensions fournies pour leur envoyer au script python qui génère une videos à partir du code extrait et via les dimensions fournit.