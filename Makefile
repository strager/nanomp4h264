CC = cc
CFLAGS = -O3 -g
LDFLAGS = -lm

EXAMPLES = examples/octagrams examples/octagrams_opencl

.PHONY: all
all: $(EXAMPLES)

examples/octagrams: examples/octagrams.c nanomp4h264.c nanomp4h264.h
	@echo building: $(@)
	@$(CC) $(CFLAGS) -o $(@) examples/octagrams.c nanomp4h264.c $(LDFLAGS)

examples/octagrams_opencl: examples/octagrams_opencl.c nanomp4h264.c nanomp4h264.h
	@echo building: $(@)
	@$(CC) $(CFLAGS) -o $(@) examples/octagrams_opencl.c nanomp4h264.c $(LDFLAGS) -framework OpenCL

.PHONY: clean
clean:
	@rm -f $(EXAMPLES)
