.PHONY: test lint format app

test:
	python -m pytest -q
lint:
	ruff check src scripts tests
format:
	ruff format src scripts tests
app:
	python -m streamlit run app/Home.py
