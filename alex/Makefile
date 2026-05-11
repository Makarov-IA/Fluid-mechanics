CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2
CPPFLAGS += -I. -I../External_libs

BUILD_DIR := build
TARGET := $(BUILD_DIR)/solver_app.exe
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
FORCING_LIB := $(BUILD_DIR)/libomega_forcing.dylib
SHARED_FLAGS := -dynamiclib -install_name @rpath/libomega_forcing.dylib
RPATH_FLAGS := -Wl,-rpath,@loader_path
else
FORCING_LIB := $(BUILD_DIR)/libomega_forcing.so
SHARED_FLAGS := -shared
RPATH_FLAGS := -Wl,-rpath,'$$ORIGIN'
endif
SOURCES := main.cpp solver/solver.cpp
HEADERS := solver/solver.hpp solver/solver_utils.hpp utils/config.hpp forcing/omega_forcing.hpp
FORCING_SOURCE := forcing/omega_forcing.cpp

.PHONY: all clean run

all: $(FORCING_LIB) $(TARGET)

$(FORCING_LIB): $(FORCING_SOURCE) forcing/omega_forcing.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -fPIC $(SHARED_FLAGS) $(FORCING_SOURCE) -o $(FORCING_LIB)

$(TARGET): $(SOURCES) $(HEADERS) $(FORCING_LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(SOURCES) -L$(BUILD_DIR) -lomega_forcing $(RPATH_FLAGS) -o $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(BUILD_DIR)/solver_app $(BUILD_DIR)/libomega_forcing.dylib $(BUILD_DIR)/libomega_forcing.so
