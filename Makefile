CC	=	g++
CFLAGS	=	-ansi -pedantic -Wall -g
LIBS    =  `pkg-config --libs opencv`  
LIBGL   =  -larmadillo -lGL -lGLU -lglut -L/usr/local/lib/x86_64-linux-gnu
INCS    =  -I /usr/local/include 

EXECUTABLES = ws3D display displayMIP displayWS autows3D  
OBJECT_FILES=flu_fun.o handlers.o imgproc3D.o GL.o 
HPP_FILES=include/constants.hpp
all: $(EXECUTABLES)

$(EXECUTABLES):	% : %.c $(OBJECT_FILES) $(HPP_FILES)
	$(CC) $(CFLAGS) -o $@ $< $(OBJECT_FILES) $(LIBS) $(LIBGL) $(INCS) 

$(OBJECT_FILES): %.o : %.cpp $(HPP_FILES) 
	$(CC) $(CFLAGS) -c $< 

clean:
	rm $(EXECUTABLES)
	rm *.o


