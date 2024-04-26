import typer
import streamlit as st

from dotenv import load_dotenv


def main(env_path: str):
    load_dotenv(env_path)
    