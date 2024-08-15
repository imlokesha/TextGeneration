# Commands


## To create conda environment

```
conda create -n env_name python==3.7
conda create --prefix ./env python=3.7 -y
```


## configure your conda environment in pycharm

```buildoutcfg
conda activate env_name
conda activate ./env
```


## Install requirements.txt
```buildoutcfg
pip install -r requirements.txt
```


### to create requirements.txt
```buildoutcfg
pip freeze>requirements.txt
```

## download gitignore using curl

```bash
curl https://raw.githubusercontent.com/c17hawke/general_template/main/.gitignore > .gitignore
```
## download init_setup.sh using curl

```bash
curl https://raw.githubusercontent.com/c17hawke/general_template/main/init_setup.sh > init_setup.sh
```
## tensorflow verification

```bash
python -c "import tensorflow as tf;print(tf.config.list_physical_devices('GPU'))"
```