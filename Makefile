install-dependencies:
	pip install -r requirements.txt
	python dep.py
	python -m spacy download en_core_web_sm

