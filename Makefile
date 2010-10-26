
ARBB_DIR=/opt/intel/arbb/1.0.0.008

CPPFLAGS=-I$(ARBB_DIR)/include
LDFLAGS=-L$(ARBB_DIR)/lib/intel64 -larbb -ltbb -lpthread

all: hello

hello: hello.o

clean:
	rm hello
	rm -f *.o
