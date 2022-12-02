all: initialize example visualize
run: example visualize

clean:
	@echo "Cleaning results folder...."
	rm ./results/*
	@echo "Cleaning figures folder...."
	rm ./figures/*

example:
	@echo "Running example.py...."
	python3 scripts/example.py

visualize:
	@echo "Running visualize.py...."
	python3 scripts/visualize.py

initialize:
	@echo "Creating virtualenv...."
	python3 -m venv venv
	@echo "Activating venv...."
	. ./venv/bin/activate
	@echo "Upgrading pip3"
	python3 -m pip install --upgrade pip
	@echo "Downloading packages...."
	pip3 install -r requirements.txt
	@echo "Creating result folders...."
	mkdir figures
	mkdir figures/gif
	mkdir results
