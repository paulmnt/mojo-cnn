
# Define V=1 for a more verbose compilation
ifndef V
        QUIET_CXX           = @echo '   ' CXX $@;
        QUIET_MKDIR         = @echo '   ' MKDIR $@;
        QUIET_RUN           = @echo '   ' RUN $@;
        QUIET_WGET           = @echo '   ' WGET $@;
        QUIET_CLEAN         = @echo '   ' CLEAN $@;
endif

RM = rm -rf

OPENCV ?= /opt/opencv

INCDIR = -I$(OPENCV)/include -I../mojo/

CXXFLAGS ?= -std=c++11 -O3
CXXFLAGS += $(INCDIR)
ifdef OMP
CXXFLAGS += -fopenmp -DMOJO_OPM
endif
ifdef OCV2
CXXFLAGS += -DMOJO_CV2
endif
ifdef OCV3
CXXFLAGS += -DMOJO_CV3
endif
ifdef AVX
CXXFLAGS += -DMOJO_AVX -msse4 -mavx
endif

LDLIBS = -L$(OPENCV)/lib

LDFLAGS = \
	-lopencv_videostab \
	-lopencv_photo \
	-lopencv_stitching \
	-lopencv_objdetect \
	-lopencv_video \
	-lopencv_ml \
	-lopencv_calib3d \
	-lopencv_features2d \
	-lopencv_imgcodecs \
	-lopencv_highgui \
	-lopencv_flann \
	-lopencv_imgproc \
	-lopencv_core

SRCS := test.cpp train_mnist.cpp train_cifar.cpp vgg.cpp
OBJS := $(SRCS:.cpp=)
HDRS := $(wildcard ../mojo/*.h)

all: $(OBJS)

.SUFFIX: .c .cpp

.c:
	$(QUIET_CXX) $(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS) $(LDFLAGS)

clean:
	$(QUIET_CLEAN) $(RM) $(OBJS)

distclean: clean
	$(QUIET_CLEAN) $(RM) ../data/cifar-10-batches-bin ../data/cifar-10-binary.tar.gz ../data/mnist

.PHONY: all clean distclean



MNIST_DATA := ../data/mnist/train-labels-idx1-ubyte \
		../data/mnist/t10k-labels-idx1-ubyte \
		../data/mnist/train-images-idx3-ubyte \
		../data/mnist/t10k-images-idx3-ubyte

$(MNIST_DATA):
	$(QUIET_MKDIR) mkdir -p ../data/mnist
	$(QUIET_WGET) wget $(subst ../data,http://yann.lecun.com/exdb,$@.gz) -O $@.gz
	gzip -d $@.gz

../data/cifar-10-binary.tar.gz:
	wget $(subst ../data,https://www.cs.toronto.edu/~kriz/, $@) -O $@


../data/cifar-10-batches-bin: ../data/cifar-10-binary.tar.gz
	tar xf $< -C ../data


data: $(MNIST_DATA) ../data/cifar-10-batches-bin

.PHONY: data

#TODO: write run targets with LD_LIBRARY_PATH set and data dependencies
