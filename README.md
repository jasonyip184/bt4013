# bt4013
**create an environment named bt4013_env from the .yml file if you don't have the bt4013_env locally yet**
conda env create -f bt4013_env.yml

**export your environment if you have new packages and ready to push**
activate bt4013_env
conda env export > bt4013_env.yml

**activate your bt4013_env locally if you have it, then update the latest packages from the new .yml file after pulling**
conda env update --name bt4013_env --file bt4013_env.yml  --prune