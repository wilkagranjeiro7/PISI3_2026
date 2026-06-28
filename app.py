"""Ponto de entrada do dashboard.

Execute com python app.py a partir da raiz do projeto.
"""

from Dashboard.app import app, run_app, server

__all__ = ["app", "server"]


if __name__ == "__main__":
    run_app()
