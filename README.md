# Setup
- Create a dotenv `.env` file that holds environment variables, 
required environment variables can be found in [`Settings`](src/core/utils/settings.py)
# CLI
- To run scripts in the [`cli`](src/cli) folder, please run with python module. Example:
```commandline
python -m src.cli.embedding args
```
- To understand how to run each script, please use `--help`. Example:
```commandline
python -m src.cli.embedding --help
```