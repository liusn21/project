CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread
LDFLAGS = -lpcap

TARGET = pcap_flow_filter
SRC = pcap_flow_filter.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean