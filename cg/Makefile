
ARBB_DIR=/opt/intel/arbb/1.0.0.008

CPPFLAGS=-I$(ARBB_DIR)/include
CXXFLAGS=-O2
LDFLAGS=-L$(ARBB_DIR)/lib/intel64 -larbb -ltbb -lpthread

OBJS=sparse.o cg.o sgs.o laplace.o util.o

all: cg cg_orig cg_sliced ax ax_orig

cg : $(OBJS) main.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cg_sliced : $(OBJS) main_sliced.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

cg_orig : $(OBJS) main_orig.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

ax : $(OBJS) main_ax.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

ax_orig : $(OBJS) main_ax_orig.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f cg cg_sliced cg_orig ax
	rm -f *.o
