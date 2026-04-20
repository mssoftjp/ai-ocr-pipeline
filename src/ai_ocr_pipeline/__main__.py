"""Module entrypoint for ``python -m ai_ocr_pipeline``."""

from ai_ocr_pipeline.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
