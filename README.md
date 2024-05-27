# Setup
- Create a dotenv `.env` file that holds environment variables named `local.env`, 
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
# Streamlit demo
- You can start the Streamlit web server using the command 
```commandline
python -m streamlit run src/core/streamlit/main.py
```