CC = cc
CFLAGS = -O3 -g
LDFLAGS = -lm

EXAMPLES = examples/octagrams examples/octagrams_opencl examples/tunnel

.PHONY: all
all: $(EXAMPLES)

examples/octagrams: examples/octagrams.c nanomp4h264.c nanomp4h264.h
	@echo building: $(@)
	@$(CC) $(CFLAGS) -o $(@) examples/octagrams.c nanomp4h264.c $(LDFLAGS)

examples/octagrams_opencl: examples/octagrams_opencl.c nanomp4h264.c nanomp4h264.h
	@echo building: $(@)
	@$(CC) $(CFLAGS) -o $(@) examples/octagrams_opencl.c nanomp4h264.c $(LDFLAGS) -framework OpenCL

examples/tunnel: examples/tunnel.c nanomp4h264.c nanomp4h264.h
	@echo building: $(@)
	@$(CC) $(CFLAGS) -o $(@) examples/tunnel.c nanomp4h264.c $(LDFLAGS)

.PHONY: clean
clean:
	@rm -f $(EXAMPLES)
