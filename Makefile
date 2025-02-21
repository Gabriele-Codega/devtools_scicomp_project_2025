.PHONY: make

make: 
	python -m pip install -r requirements.txt
	python src/pyclassify/_compile.py
	python -m pip install -e .
