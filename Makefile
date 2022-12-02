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
	@echo "Creating result folders...."
	mkdir figures
	mkdir figures/gif
	mkdir results
