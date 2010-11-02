
ARBB_DIR=/opt/intel/arbb/1.0.0.008

CPPFLAGS=-I$(ARBB_DIR)/include
CXXFLAGS=-O2
LDFLAGS=-L$(ARBB_DIR)/lib/intel64 -larbb -ltbb -lpthread

OBJS=sparse.o cg.o sgs.o laplace.o main.o

all: cg

cg : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cg
	rm -f *.o
