
####################################################
#                                                  # 
# Universidade de São Paulo                        # 
# Instituto de Matemática e Estatística - IME-USP  # 
#												   #	
# Mestrado em Ciência da Computação                #
# William Lira Ferreira                            #            
#                                                  #
####################################################

# Configurations
#
CC = g++
LD = nvcc
#WARNINGS = -Wall -Wno-unused-variable
WARNINGS =

LIBRARY_LDFLAGS = -lc -shared -Xcompiler -fPIC

#OPTIMIZED(O2)#
# No pegrande alterar arch=sm_13 
CFLAGS = -O3 $(WARNINGS)
LDFLAGS = --ptxas-options=-v -arch=sm_12 -use_fast_math -Xcompiler -fpic -Xcompiler -fopenmp -lgomp -Xcompiler -ffloat-store -Xcompiler -fexpensive-optimizations 
#--ptxas-options=-v -arch=sm_12 -use_fast_math -Xcompiler -fopenmp -lgomp 
#

#DEGUB INFO#
#CFLAGS = -g
#LDFLAGS = --ptxas-options=-v -arch=sm_12 -use_fast_math -Xcompiler -fpic -Xcompiler -fopenmp -lgomp

SOURCE_ROOT 	= src
TEST_SOURCE_ROOT = tests

BUILD_ROOT 		= build
MODULES 	    = core kernel javanative
LIBRARY_NAME = libbioinfogen.so
EXECUTABLE_NAME = BioinfoGeneInteractions

LIBRARY	= $(BUILD_ROOT)/$(LIBRARY_NAME)
EXECUTABLE	= $(BUILD_ROOT)/$(EXECUTABLE_NAME)
SOURCE_DIRS = $(addprefix $(SOURCE_ROOT)/, $(MODULES))
BUILD_DIRS	= $(addprefix $(BUILD_ROOT)/, $(MODULES))

SOURCES 	 = $(foreach sdir, $(SOURCE_DIRS), $(wildcard $(sdir)/*.cpp) )
CUDA_SOURCES = $(foreach sdir, $(SOURCE_DIRS), $(wildcard $(sdir)/*.cu) )
OBJECTS 	 = $(patsubst src/%.cpp, build/%.o, $(SOURCES))
INCLUDES     = $(addprefix -I, $(SOURCE_DIRS))

# Include JAVAC
JAVA_INCLUDE = /usr/local/bin/jdk1.6.0_22/include
#JAVA_INCLUDE = /usr/lib/jvm/java-1.6.0-openjdk/include
JAVA_INCLUDE_LINUX = /usr/local/bin/jdk1.6.0_22/include/linux
#JAVA_INCLUDE_LINUX = /usr/lib/jvm/java-1.6.0-openjdk/include/linux
INCLUDE_C       = $(addprefix -I, $(JAVA_INCLUDE) $(JAVA_INCLUDE_LINUX))


# Execution Targets

all: checkdirs $(SOURCES) $(CUDA_SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	@echo ' '
	@echo 'Building Shared Library: $(LIBRARY_NAME)'
	$(LD) -o $(LIBRARY) $(CFLAGS) $(LDFLAGS) $(LIBRARY_LDFLAGS) $(INCLUDE_C) $(INCLUDES) $(SOURCES) $(CUDA_SOURCES)
	
	@echo ' '
	@echo 'Building Executable: $@'
	$(LD) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(CUDA_SOURCES) $(OBJECTS) -o $@
	@echo ' '
	@echo 'Building Finished...'
	@echo ' '

build/%.o: src/%.cpp
#	$(CC) $(CFLAGS) $(INCLUDE_C) $(INCLUDES) $< -o $@
	$(CC) $(CFLAGS) $(INCLUDE_C) $(INCLUDES) $< -c -o $@
	
	
# Make build diretories
checkdirs: $(BUILD_DIRS)
	
$(BUILD_DIRS):
	@mkdir -p $@

# Clean build diretories and Executable
clean:
	rm -rf $(BUILD_DIRS) $(EXECUTABLE) $(LIBRARY) $(OBJECTS)

tests:

