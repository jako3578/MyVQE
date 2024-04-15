Out.txt: main.py
	python3 main.py > /dev/null

clean:
	rm -f Out.txt