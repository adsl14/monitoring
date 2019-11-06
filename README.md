# monitoring

1. Instalar Anaconda : https://www.anaconda.com/distribution/?gclid=EAIaIQobChMIopvCio3R5QIV05rVCh1Isg3qEAAYASAAEgKoQfD_BwE#download-section

2. Crear un entorno virtual
2.1 Acceder a Anaconda prompt
2.2 Escribir "conda create -n ["environment-name"]. Ejemplo: conda create -n tensor

3. Instalar tensorflow
3.1 Acceder al entorno virtual -> activate tensor
3.2 conda install -c conda-forge tensorflow

4. Instalar keras
4.1 conda install -c conda-forge keras

5. Instalar plaidml (solo para tarjetas AMD)
5.1 pip install plaidml-keras plaidbench
5.2 plaidml-setup

6. Instalar folium
6.1 conda install -c conda-forge folium

7. Instalar Earth Engine API
7.1 conda install -c conda-forge earthengine-api

IMPORTANTE:

Si utilizas gráfica AMD y has instalado plaidml, poner al principio del código lo siguiente: import plaidml.keras
plaidml.keras.install_backend()
