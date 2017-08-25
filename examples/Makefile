
#
# Aliases
#

# Define V=1 for a more verbose compilation
ifndef V
        QUIET_CXX           = @echo '   ' CXX $@;
        QUIET_MKDIR         = @echo '   ' MKDIR $@;
        QUIET_RUN           = @echo '   ' RUN $@;
        QUIET_UNZIP         = @echo '   ' UNZIP $@;
        QUIET_WGET          = @echo '   ' WGET $@;
        QUIET_CLEAN         = @echo '   ' CLEAN $@;
endif

RM = rm -rf


#
# Environment variables
#

OPENCV ?= /opt/opencv


#
# Compile
#

INCDIR = -I$(OPENCV)/include -I../mojo/

CXXFLAGS ?= -std=c++11 -O3
CXXFLAGS += $(INCDIR)
ifdef OMP
CXXFLAGS += -fopenmp -DMOJO_OMP
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
HDRS := $(wildcard ../mojo/*.h) $(wildcard ./*.h)

all: $(OBJS)

.SUFFIX: .c .cpp

# The following rules are inferred by make. Overriding to add $(QUIET_CXX)
.cpp:
	$(QUIET_CXX) $(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS) $(LDFLAGS)

$(OBJS): $(HDRS)

clean:
	$(QUIET_CLEAN) $(RM) $(OBJS)

distclean: clean
	$(QUIET_CLEAN) $(RM) \
	../data/cifar-10-batches-bin \
	../data/cifar-10-binary.tar.gz \
	../data/mnist \
	$(MOJO_PRETRAINED) \
	../models/snapshots/*.htm \
	../models/snapshots/*.txt

.PHONY: all clean distclean


#
# Get Data Sets and Pretrained Models
#

MNIST_DATA := ../data/mnist/train-labels-idx1-ubyte \
		../data/mnist/t10k-labels-idx1-ubyte \
		../data/mnist/train-images-idx3-ubyte \
		../data/mnist/t10k-images-idx3-ubyte

MOJO_PRETRAINED := ../models/cifar_deepcnet.mojo \
		../models/mnist_deepcnet.mojo \
		../models/vgg16.mojo

$(MNIST_DATA):
	$(QUIET_MKDIR) mkdir -p ../data/mnist
	$(QUIET_WGET) wget $(subst ../data,http://yann.lecun.com/exdb,$@.gz) -O $@.gz
	$(QUIET_UNZIP) gzip -d $@.gz

../data/cifar-10-binary.tar.gz:
	$(QUIET_WGET) wget $(subst ../data,https://www.cs.toronto.edu/~kriz/, $@) -O $@


../data/cifar-10-batches-bin: ../data/cifar-10-binary.tar.gz
	$(QUIET_UNZIP) tar xf $< -C ../data
	@touch $@

$(MOJO_PRETRAINED):
	$(QUIET_WGET) wget $(subst ../models,http://espdev.cs.columbia.edu/data_sets,$@) -O $@

data: $(MNIST_DATA) ../data/cifar-10-batches-bin $(MOJO_PRETRAINED)

.PHONY: data


#
# Run
#

RUNS := $(SRCS:.cpp=-run)

$(RUNS): %-run : %-data %
	@LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):$(OPENCV)/lib ./$(@:-run=)

test-data: ../data/mnist/t10k-labels-idx1-ubyte \
	../data/mnist/t10k-images-idx3-ubyte \
	../models/cifar_deepcnet.mojo \
	../models/mnist_deepcnet.mojo

vgg-data: ../data/dog.jpg ../models/vgg16.mojo

train_mnist-data: $(MNIST_DATA)

train_cifar-data:  ../data/cifar-10-batches-bin

.PHONY: $(RUNS)
