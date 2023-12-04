# clinicadl-pythae
Plugin to use Pythae models with ClinicaDL


## some informations

If not working on cluster 

1- First, `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` will work for following poetry commands until you close (exit) your shell session
2- Add an environment variable for each! poetry command, for example, `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring poetry install`
3- If you want to preserve (store) this environment variable between shell sessions or system reboots you can add it in `.bashrc` and `.profile`

example for bash shell:
```
echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.profile
exec "$SHELL"
```
for case number 3) you can now run any poetry command as usual, even after system restart

## IN CONSTRUCTION 

