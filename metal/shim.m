// C ABI over Metal for cl-gpu. No ARC: constructors return +1 objects and clgpu_release frees them.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdlib.h>
#include <string.h>

enum { CLGPU_BUFFER = 0, CLGPU_BYTES = 1, CLGPU_THREADGROUP = 2 };

static char *copy_string(NSString *string) {
    return string ? strdup([string UTF8String]) : NULL;
}

static void set_error(char **error, NSString *message) {
    if (error)
        *error = copy_string(message);
}

void clgpu_free_string(char *string) { free(string); }

void clgpu_release(void *object) {
    if (object)
        [(id)object release];
}

void *clgpu_device_default(void) {
    @autoreleasepool {
        return (void *)MTLCreateSystemDefaultDevice();
    }
}

char *clgpu_device_name(void *device) {
    @autoreleasepool {
        return copy_string([(id<MTLDevice>)device name]);
    }
}

unsigned long clgpu_device_max_threads(void *device) {
    MTLSize size = [(id<MTLDevice>)device maxThreadsPerThreadgroup];
    return size.width;
}

unsigned long clgpu_device_max_threadgroup_memory(void *device) {
    return [(id<MTLDevice>)device maxThreadgroupMemoryLength];
}

void *clgpu_queue_new(void *device) {
    return (void *)[(id<MTLDevice>)device newCommandQueue];
}

void *clgpu_buffer_new(void *device, unsigned long length) {
    return (void *)[(id<MTLDevice>)device newBufferWithLength:length
                                                      options:MTLResourceStorageModeShared];
}

void *clgpu_buffer_contents(void *buffer) { return [(id<MTLBuffer>)buffer contents]; }

unsigned long clgpu_buffer_length(void *buffer) { return [(id<MTLBuffer>)buffer length]; }

void *clgpu_library_new(void *device, const char *source, char **error) {
    @autoreleasepool {
        NSError *failure = nil;
        NSString *text = [NSString stringWithUTF8String:source];
        id<MTLLibrary> library = [(id<MTLDevice>)device newLibraryWithSource:text
                                                                     options:nil
                                                                       error:&failure];
        if (!library)
            set_error(error, failure ? [failure localizedDescription] : @"library compilation failed");
        return (void *)library;
    }
}

void *clgpu_pipeline_new(void *device, void *library, const char *name, char **error) {
    @autoreleasepool {
        NSError *failure = nil;
        id<MTLFunction> function =
            [(id<MTLLibrary>)library newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!function) {
            set_error(error, [NSString stringWithFormat:@"no kernel function named %s", name]);
            return NULL;
        }
        id<MTLComputePipelineState> pipeline =
            [(id<MTLDevice>)device newComputePipelineStateWithFunction:function error:&failure];
        [function release];
        if (!pipeline)
            set_error(error, failure ? [failure localizedDescription] : @"pipeline creation failed");
        return (void *)pipeline;
    }
}

unsigned long clgpu_pipeline_max_threads(void *pipeline) {
    return [(id<MTLComputePipelineState>)pipeline maxTotalThreadsPerThreadgroup];
}

unsigned long clgpu_pipeline_execution_width(void *pipeline) {
    return [(id<MTLComputePipelineState>)pipeline threadExecutionWidth];
}

// A stream encodes the launches of one device into a command buffer, which is committed after
// COMMIT_EVERY encodes and by clgpu_stream_wait. Compute work shares one open encoder; blit and
// MPS work end it first. Every call takes the stream's lock.
// TODO: fixed commit interval, tune or adapt to measured GPU idle time (#48)
// TODO: one lock per stream serialises encoding from many threads (#50)
#define COMMIT_EVERY 64

@interface CLGPUStream : NSObject {
@public
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> command;
    id<MTLComputeCommandEncoder> encoder;
    unsigned long encoded;
    NSMutableArray *pending;
}
@end

@implementation CLGPUStream
- (void)dealloc {
    [encoder endEncoding];
    [encoder release];
    [command release];
    [queue release];
    [pending release];
    [super dealloc];
}
@end

void *clgpu_stream_new(void *queue) {
    CLGPUStream *stream = [[CLGPUStream alloc] init];
    stream->queue = [(id<MTLCommandQueue>)queue retain];
    stream->pending = [[NSMutableArray alloc] init];
    return (void *)stream;
}

static void end_encoder(CLGPUStream *stream) {
    if (stream->encoder) {
        [stream->encoder endEncoding];
        [stream->encoder release];
        stream->encoder = nil;
    }
}

static id<MTLCommandBuffer> open_command(CLGPUStream *stream) {
    if (!stream->command)
        stream->command = [[stream->queue commandBuffer] retain];
    return stream->command;
}

static void commit(CLGPUStream *stream) {
    end_encoder(stream);
    if (!stream->command)
        return;
    [stream->command commit];
    [stream->pending addObject:stream->command];
    [stream->command release];
    stream->command = nil;
    stream->encoded = 0;
    for (NSUInteger i = 0; i < [stream->pending count];) {
        id<MTLCommandBuffer> older = [stream->pending objectAtIndex:i];
        if ([older status] == MTLCommandBufferStatusCompleted)
            [stream->pending removeObjectAtIndex:i];
        else
            i++;
    }
}

static void note_encoded(CLGPUStream *stream) {
    if (++stream->encoded >= COMMIT_EVERY)
        commit(stream);
}

// Arguments are bound in order. KINDS[i] selects how VALUES[i] and LENGTHS[i] are used:
// a buffer, bytes copied from the pointer, or a threadgroup allocation of LENGTHS[i].
// The buffer index and the threadgroup index count separately, as in the generated source.
// Returns 0 on success.
int clgpu_stream_dispatch(void *handle, void *pipeline, int count, const int *kinds,
                          void *const *values, const unsigned long *lengths,
                          const unsigned long *grid, const unsigned long *group, char **error) {
    CLGPUStream *stream = (CLGPUStream *)handle;
    @autoreleasepool {
        @synchronized(stream) {
            if (!stream->encoder) {
                id<MTLCommandBuffer> command = open_command(stream);
                stream->encoder = [[command computeCommandEncoder] retain];
                if (!stream->encoder) {
                    set_error(error, @"could not create a command encoder");
                    return 1;
                }
            }
            id<MTLComputeCommandEncoder> encoder = stream->encoder;
            [encoder setComputePipelineState:(id<MTLComputePipelineState>)pipeline];
            NSUInteger buffers = 0, groups = 0;
            for (int i = 0; i < count; i++) {
                switch (kinds[i]) {
                case CLGPU_BUFFER:
                    [encoder setBuffer:(id<MTLBuffer>)values[i] offset:0 atIndex:buffers++];
                    break;
                case CLGPU_BYTES:
                    [encoder setBytes:values[i] length:lengths[i] atIndex:buffers++];
                    break;
                case CLGPU_THREADGROUP:
                    [encoder setThreadgroupMemoryLength:lengths[i] atIndex:groups++];
                    break;
                }
            }
            [encoder dispatchThreads:MTLSizeMake(grid[0], grid[1], grid[2])
               threadsPerThreadgroup:MTLSizeMake(group[0], group[1], group[2])];
            note_encoded(stream);
            return 0;
        }
    }
}

// Commit and wait for everything encoded so far. Returns 1 with the message of the first command
// that failed since the last call, else 0.
int clgpu_stream_wait(void *handle, char **error) {
    CLGPUStream *stream = (CLGPUStream *)handle;
    @autoreleasepool {
        NSArray *waiting;
        @synchronized(stream) {
            commit(stream);
            waiting = [[NSArray alloc] initWithArray:stream->pending];
        }
        for (id<MTLCommandBuffer> command in waiting)
            [command waitUntilCompleted];
        int status = 0;
        @synchronized(stream) {
            for (id<MTLCommandBuffer> command in waiting) {
                if (![stream->pending containsObject:command])
                    continue;
                [stream->pending removeObject:command];
                if (!status && [command status] == MTLCommandBufferStatusError) {
                    NSError *failure = [command error];
                    set_error(error, failure ? [failure localizedDescription] : @"command failed");
                    status = 1;
                }
            }
        }
        [waiting release];
        return status;
    }
}

// MetalPerformanceShaders. A matrix is described by five values: byte offset, rows, columns, row
// bytes and, for a batch, matrix bytes. Both calls encode onto a stream and return 0 on success.
// MPS rejects some bad arguments with an assertion that aborts the process, so the caller passes
// only shapes it has checked.

int clgpu_mps_supported(void *device) {
    return MPSSupportsMTLDevice((id<MTLDevice>)device) ? 1 : 0;
}

static MPSMatrix *new_matrix(void *buffer, const unsigned long *shape, unsigned long count) {
    MPSMatrixDescriptor *descriptor =
        count == 1 ? [MPSMatrixDescriptor matrixDescriptorWithRows:shape[1]
                                                           columns:shape[2]
                                                          rowBytes:shape[3]
                                                          dataType:MPSDataTypeFloat32]
                   : [MPSMatrixDescriptor matrixDescriptorWithRows:shape[1]
                                                           columns:shape[2]
                                                          matrices:count
                                                          rowBytes:shape[3]
                                                       matrixBytes:shape[4]
                                                          dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)buffer offset:shape[0] descriptor:descriptor];
}

static MPSVector *new_vector(void *buffer, unsigned long offset, unsigned long length) {
    MPSVectorDescriptor *descriptor = [MPSVectorDescriptor vectorDescriptorWithLength:length
                                                                             dataType:MPSDataTypeFloat32];
    return [[MPSVector alloc] initWithBuffer:(id<MTLBuffer>)buffer offset:offset descriptor:descriptor];
}

// MPS objects the command buffer needs live until it completes.
static void keep_alive(id<MTLCommandBuffer> command, NSArray *objects) {
    [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        (void)finished;
        (void)objects;
    }];
}

// MPS reads the result even when BETA is 0, so a NaN there would survive. Clear it first.
static void clear_bytes(id<MTLCommandBuffer> command, void *buffer, unsigned long offset,
                        unsigned long rows, unsigned long row_length, unsigned long row_bytes,
                        unsigned long count, unsigned long matrix_bytes) {
    id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
    if (row_length == row_bytes && (count == 1 || matrix_bytes == rows * row_bytes)) {
        [blit fillBuffer:(id<MTLBuffer>)buffer
                   range:NSMakeRange(offset, count * rows * row_length)
                   value:0];
    } else {
        for (unsigned long product = 0; product < count; product++)
            for (unsigned long row = 0; row < rows; row++)
                [blit fillBuffer:(id<MTLBuffer>)buffer
                           range:NSMakeRange(offset + product * matrix_bytes + row * row_bytes,
                                             row_length)
                           value:0];
    }
    [blit endEncoding];
}

// One encode covers at most this many products; MPS leaves the rest unwritten.
#define MPS_MAX_BATCH 65535

// RESULT = ALPHA * op(LEFT) * op(RIGHT) + BETA * RESULT, with RESULT of M by N and K columns inside,
// for each of COUNT products. BUFFERS and SHAPES hold the left, right and result matrices in that
// order.
int clgpu_mps_gemm(void *handle, int transpose_left, int transpose_right,
                   unsigned long m, unsigned long n, unsigned long k, float alpha, float beta,
                   unsigned long count, void *const *buffers, const unsigned long *shapes,
                   char **error) {
    CLGPUStream *stream = (CLGPUStream *)handle;
    @autoreleasepool {
        @synchronized(stream) {
            @try {
                NSMutableArray *kept = [NSMutableArray array];
                MPSMatrixMultiplication *multiplication =
                    [[[MPSMatrixMultiplication alloc] initWithDevice:[stream->queue device]
                                                       transposeLeft:transpose_left
                                                      transposeRight:transpose_right
                                                          resultRows:m
                                                       resultColumns:n
                                                     interiorColumns:k
                                                               alpha:alpha
                                                                beta:beta] autorelease];
                [kept addObject:multiplication];
                end_encoder(stream);
                id<MTLCommandBuffer> command = open_command(stream);
                if (beta == 0)
                    clear_bytes(command, buffers[2], shapes[10], shapes[11], shapes[12] * 4,
                                shapes[13], count, shapes[14]);
                for (unsigned long start = 0; start < count; start += MPS_MAX_BATCH) {
                    unsigned long chunk = MIN(MPS_MAX_BATCH, count - start);
                    MPSMatrix *matrices[3];
                    for (int i = 0; i < 3; i++) {
                        unsigned long shape[5];
                        memcpy(shape, shapes + 5 * i, sizeof shape);
                        shape[0] += start * shape[4];
                        matrices[i] = [new_matrix(buffers[i], shape, chunk) autorelease];
                        [kept addObject:matrices[i]];
                    }
                    [multiplication encodeToCommandBuffer:command
                                               leftMatrix:matrices[0]
                                              rightMatrix:matrices[1]
                                             resultMatrix:matrices[2]];
                }
                keep_alive(command, kept);
                note_encoded(stream);
                return 0;
            } @catch (NSException *exception) {
                set_error(error, [exception reason]);
                return 1;
            }
        }
    }
}

// RESULT = ALPHA * op(MATRIX) * INPUT + BETA * RESULT, where op(MATRIX) is ROWS by COLUMNS.
// BUFFERS holds the matrix, input and result vector; SHAPES the matrix's five values, then the
// byte offset and length of the input and of the result.
int clgpu_mps_gemv(void *handle, int transpose, unsigned long rows, unsigned long columns,
                   float alpha, float beta, void *const *buffers, const unsigned long *shapes,
                   char **error) {
    CLGPUStream *stream = (CLGPUStream *)handle;
    @autoreleasepool {
        @synchronized(stream) {
            @try {
                MPSMatrix *matrix = [new_matrix(buffers[0], shapes, 1) autorelease];
                MPSVector *input = [new_vector(buffers[1], shapes[5], shapes[6]) autorelease];
                MPSVector *result = [new_vector(buffers[2], shapes[7], shapes[8]) autorelease];
                MPSMatrixVectorMultiplication *multiplication =
                    [[[MPSMatrixVectorMultiplication alloc] initWithDevice:[stream->queue device]
                                                                 transpose:transpose
                                                                      rows:rows
                                                                   columns:columns
                                                                     alpha:alpha
                                                                      beta:beta] autorelease];
                end_encoder(stream);
                id<MTLCommandBuffer> command = open_command(stream);
                if (beta == 0)
                    clear_bytes(command, buffers[2], shapes[7], 1, shapes[8] * 4, shapes[8] * 4, 1, 0);
                [multiplication encodeToCommandBuffer:command
                                          inputMatrix:matrix
                                          inputVector:input
                                         resultVector:result];
                keep_alive(command, @[matrix, input, result, multiplication]);
                note_encoded(stream);
                return 0;
            } @catch (NSException *exception) {
                set_error(error, [exception reason]);
                return 1;
            }
        }
    }
}
