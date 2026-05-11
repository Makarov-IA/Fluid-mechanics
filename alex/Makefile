CXX ?= g++

.PHONY: all clean run cpu

all cpu:
	$(MAKE) -C cpu all CXX="$(CXX)"

run:
	$(MAKE) -C cpu run CXX="$(CXX)"

clean:
	$(MAKE) -C cpu clean CXX="$(CXX)"
