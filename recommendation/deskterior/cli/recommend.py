"""Setup recommender CLI entrypoint.

Preferred command:
    python -m deskterior.cli.recommend
"""

from deskterior.recommender.engine import main


if __name__ == "__main__":
    main()
