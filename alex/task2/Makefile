CXX = g++
CXXFLAGS = -O3 -std=c++17 -march=native

ifeq ($(OS),Windows_NT)
    CXXFLAGS += -mstackrealign
    EXE = .exe
else
    EXE =
endif

TARGET = task2$(EXE)
SOURCES = main.cpp solver.cpp
HEADERS = solver.hpp
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)

run-test: $(TARGET)
	./$(TARGET) --Nx 51 --Ny 51 --Nt 100 --nu 0.01 --lid 1 --forcing sin --forcing-amp 0.5 --forcing-omega 6.283185307 --save-every 25 --output-dir results_test

test: $(TARGET)
	bash ./scripts/test.sh

plot:
	bash ./scripts/plot.sh --input-dir results --output-dir plots --field omega --graph all --snapshot-index -1 --save --no-show

clean:
	rm -f $(TARGET) $(OBJECTS)

clean-results:
	rm -rf results results_test plots plots_test tmp_test_runs

help:
	@echo "make all                      - build solver"
	@echo "make run ARGS='...'           - run with custom args"
	@echo "make run-test                 - sample run"
	@echo "make test                     - automated tests"
	@echo "make plot                     - sample plot"
	@echo "make clean                    - remove build artifacts"
	@echo "make clean-results            - remove generated data"

.PHONY: all run run-test test plot clean clean-results help
