# Monitoring
-------------

Machine learning network for monitoring areas.

## 1. Installing tensorflow, keras, earth engine, etc.

### 1.1 Install Anaconda : [Anaconda](https://www.anaconda.com/distribution/?gclid=EAIaIQobChMIopvCio3R5QIV05rVCh1Isg3qEAAYASAAEgKoQfD_BwE#download-section)

### 1.2. Create virtual environment
- Execute Anaconda prompt by searching it (Windows), or by writting this in a Linux's terminal: `source anaconda3/bin/activate`
- In the terminal, write `conda create -n 'environment-name'`. 
	- For example: `conda create -n tensor`

###	1.3. Install tensorflow 1.15
- Activate the virtual environment that you has created before: `activate tensor`
-	Install tensorflow: `conda install -c conda-forge tensorflow==1.15`

### 1.4. Install keras
-	`conda install -c conda-forge keras`

### 1.5. Install plaidml (only for AMD cards)
-	`conda install -c conda-forge keras`
- `pip install plaidml-keras plaidbench`
- `plaidml-setup`

### 1.5. Install folium
-	`conda install -c conda-forge folium`

### 1.6. Install Earth Engine API
-	`conda install -c conda-forge earthengine-api`

***VERY IMPORTANT***:
If you're using an AMD, and has installed 'plaidml', you have to include, at the begining of the python code, the following code: `import plaidml.keras plaidml.keras.install_backend()`

## 2. Running tensorflow

### 1.1 Execute anaconda prompt by writting in a terminal: 
-	`source anaconda3/bin/activate`

### 1.2. Run the virtual environment: 
-	`conda activate tensor`