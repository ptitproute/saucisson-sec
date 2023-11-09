#! /bin/bash

recupere_cookie(){
    # récupère un cookie sur le site de shadertoy
    echo $cookie
}

extrait_OpenGl(){

local liens=("$@")
for s in "${liens[@]}"; do
openGlCode=$(curl -X POST https://www.shadertoy.com/shadertoy \
    -H 'User-Agent: Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0' \
    -H 'Accept: */*' \
    -H 'Accept-Language: fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3' \
    -H 'Accept-Encoding: gzip, deflate, br' \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -H 'Origin: https://www.shadertoy.com' \
    -H 'DNT: 1' \
    -H 'Connection: keep-alive' \
    -H "Referer: https://www.shadertoy.com/view/${s}" \
    -H 'Cookie: AWSALB=a8p3RvWthFMVR6NzETMyVqaNMe69ErE2HKS3cLthJST6AS6CUTUEBcT3XDFIcY0iyDGcnoCQlOgnmEjaA6+dZWSStcmZ2ftbVdmvxwXiSHo1kTU9NCVXVzuNw/Jb; AWSALBCORS=a8p3RvWthFMVR6NzETMyVqaNMe69ErE2HKS3cLthJST6AS6CUTUEBcT3XDFIcY0iyDGcnoCQlOgnmEjaA6+dZWSStcmZ2ftbVdmvxwXiSHo1kTU9NCVXVzuNw/Jb; sdtd=6egovnb4o5l0of1hs9ta4io1dv' \
    -H 'Sec-Fetch-Dest: empty' \
    -H 'Sec-Fetch-Mode: cors' \
    -H 'Sec-Fetch-Site: same-origin' \
    -H 'TE: trailers' \
    -d "s=%7B%20%22shaders%22%20%3A%20%5B%22${s}%22%5D%20%7D&nt=1&nl=1&np=1" 2>/dev/null | zcat |jq '.[].renderpass[].code'| sed 's/\\n/\n/g')

trim "$openGlCode" > ${DOSSIER_COURANT}/openGl/${s}.glsl
done

}
