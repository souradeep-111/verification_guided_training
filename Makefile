CXX = g++


include Makefile.locale


GUROBI_INCLUDEDIR=$(strip $(GUROBI_PATH))/$(strip $(HOST_ARCH))/include/
GUROBI_LIBDIR=$(strip $(GUROBI_PATH))/$(strip $(HOST_ARCH))/lib/


# LIBS = -lgurobi_c++ -lgurobi75 -lm -D_GLIBCXX_USE_CXX11_ABI=0 -m64 -w

LIBS = -lgurobi_c++ -lgurobi80 -lm -D_GLIBCXX_USE_CXX11_ABI=0 -m64 -w

CXXFLAGS = -MMD -I . -I ./src  -I /usr/local/include/ -I $(GUROBI_INCLUDEDIR) -g -O3 -std=c++11

LINK_FLAGS = -g -L ./ -L /usr/local/lib/ -L $(GUROBI_LIBDIR)

OBJS = ./src/propagate_intervals.o ./src/network_computation.o ./src/gurobi_interface.o ./src/configuration.o \
./src/counter_examples.o ./src/run_python.o ./src/run_verification.o
DEPENDS = ${OBJECTS:.o=.d}


all: libs verify_and_train_1 verify_and_train_2 verify_and_train_3 verify_and_train_4 verify_and_train_5\
	verify_and_train_6 verify_and_train_7 
# all: libs run_file_1 run_file_2 run_file_3 run_file_4 run_file_5 run_file_6 run_file_7 verify_and_train

libs: $(OBJS)
	ar rcs ./src/libsherlock.a $(OBJS)
	ranlib ./src/libsherlock.a
	cp ./src/*.h ./include

verify_and_train_1 : verify_and_train_1.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_2 : verify_and_train_2.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_3 : verify_and_train_3.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_4 : verify_and_train_4.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_5 : verify_and_train_5.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_6 : verify_and_train_6.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)


verify_and_train_7 : verify_and_train_7.o $(OBJS)
	$(CXX) -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) -O3 -c $(CXXFLAGS) -o $@ $< $(LIBS)
%.o: %.cpp
	$(CXX) -O3 -c $(CXXFLAGS) -o $@ $< $(LIBS)
%.o: %.c
	$(CXX) -O3 -c $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f ./src/*.o *.o ./run_file_1 ./run_file_2 ./run_file_3\
	 ./run_file_4 ./run_file_5  ./run_file_6  ./run_file_7 ./lib/* \
	./include/configuration.h ./include/gurobi_interface.h \
	./include/network_computation.h ./include/propagate_intervals.h \
	./include/counter_examples.o ./include/run_python.o ./include/run_verification.o \
	./verify_and_train_1 ./verify_and_train_2 ./verify_and_train_3 ./verify_and_train_4 \
	./verify_and_train_5 ./verify_and_train_6 ./verify_and_train_7

-include ${DEPENDS}
