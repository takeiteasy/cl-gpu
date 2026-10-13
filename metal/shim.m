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

// Arguments are bound in order. KINDS[i] selects how VALUES[i] and LENGTHS[i] are used:
// a buffer, bytes copied from the pointer, or a threadgroup allocation of LENGTHS[i].
// The buffer index and the threadgroup index count separately, as in the generated source.
void *clgpu_dispatch(void *queue, void *pipeline, int count, const int *kinds,
                     void *const *values, const unsigned long *lengths,
                     const unsigned long *grid, const unsigned long *group, char **error) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [(id<MTLCommandQueue>)queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            set_error(error, @"could not create a command encoder");
            return NULL;
        }
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
        [encoder endEncoding];
        [command commit];
        return (void *)[command retain];
    }
}

int clgpu_command_done_p(void *command) {
    MTLCommandBufferStatus status = [(id<MTLCommandBuffer>)command status];
    return status == MTLCommandBufferStatusCompleted || status == MTLCommandBufferStatusError;
}

// Returns 0 on success.
int clgpu_command_wait(void *command, char **error) {
    @autoreleasepool {
        id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>)command;
        [buffer waitUntilCompleted];
        if ([buffer status] == MTLCommandBufferStatusError) {
            NSError *failure = [buffer error];
            set_error(error, failure ? [failure localizedDescription] : @"command failed");
            return 1;
        }
        return 0;
    }
}

// MetalPerformanceShaders. A matrix is described by five values: byte offset, rows, columns, row
// bytes and, for a batch, matrix bytes. Both calls return 0 on success. MPS rejects some bad arguments with an assertion that
// aborts the process, so the caller passes only shapes it has checked.

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

static int finish(id<MTLCommandBuffer> command, char **error) {
    [command commit];
    [command waitUntilCompleted];
    if ([command status] == MTLCommandBufferStatusError) {
        NSError *failure = [command error];
        set_error(error, failure ? [failure localizedDescription] : @"command failed");
        return 1;
    }
    return 0;
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
int clgpu_mps_gemm(void *queue, int transpose_left, int transpose_right,
                   unsigned long m, unsigned long n, unsigned long k, float alpha, float beta,
                   unsigned long count, void *const *buffers, const unsigned long *shapes,
                   char **error) {
    @autoreleasepool {
        MPSMatrixMultiplication *multiplication = nil;
        int status = 1;
        @try {
            id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)queue;
            multiplication = [[MPSMatrixMultiplication alloc] initWithDevice:[commandQueue device]
                                                               transposeLeft:transpose_left
                                                              transposeRight:transpose_right
                                                                  resultRows:m
                                                               resultColumns:n
                                                             interiorColumns:k
                                                                       alpha:alpha
                                                                        beta:beta];
            id<MTLCommandBuffer> command = [commandQueue commandBuffer];
            if (beta == 0)
                clear_bytes(command, buffers[2], shapes[10], shapes[11], shapes[12] * 4, shapes[13],
                            count, shapes[14]);
            for (unsigned long start = 0; start < count; start += MPS_MAX_BATCH) {
                unsigned long chunk = MIN(MPS_MAX_BATCH, count - start);
                MPSMatrix *matrices[3];
                for (int i = 0; i < 3; i++) {
                    unsigned long shape[5];
                    memcpy(shape, shapes + 5 * i, sizeof shape);
                    shape[0] += start * shape[4];
                    matrices[i] = [new_matrix(buffers[i], shape, chunk) autorelease];
                }
                [multiplication encodeToCommandBuffer:command
                                           leftMatrix:matrices[0]
                                          rightMatrix:matrices[1]
                                         resultMatrix:matrices[2]];
            }
            status = finish(command, error);
        } @catch (NSException *exception) {
            set_error(error, [exception reason]);
        }
        [multiplication release];
        return status;
    }
}

// RESULT = ALPHA * op(MATRIX) * INPUT + BETA * RESULT, where op(MATRIX) is ROWS by COLUMNS.
// BUFFERS holds the matrix, input and result vector; SHAPES the matrix's five values, then the
// byte offset and length of the input and of the result.
int clgpu_mps_gemv(void *queue, int transpose, unsigned long rows, unsigned long columns,
                   float alpha, float beta, void *const *buffers, const unsigned long *shapes,
                   char **error) {
    @autoreleasepool {
        MPSMatrix *matrix = nil;
        MPSVector *input = nil, *result = nil;
        MPSMatrixVectorMultiplication *multiplication = nil;
        int status = 1;
        @try {
            id<MTLCommandQueue> commandQueue = (id<MTLCommandQueue>)queue;
            matrix = new_matrix(buffers[0], shapes, 1);
            input = new_vector(buffers[1], shapes[5], shapes[6]);
            result = new_vector(buffers[2], shapes[7], shapes[8]);
            multiplication = [[MPSMatrixVectorMultiplication alloc] initWithDevice:[commandQueue device]
                                                                         transpose:transpose
                                                                              rows:rows
                                                                           columns:columns
                                                                             alpha:alpha
                                                                              beta:beta];
            id<MTLCommandBuffer> command = [commandQueue commandBuffer];
            if (beta == 0)
                clear_bytes(command, buffers[2], shapes[7], 1, shapes[8] * 4, shapes[8] * 4, 1, 0);
            [multiplication encodeToCommandBuffer:command
                                      inputMatrix:matrix
                                      inputVector:input
                                     resultVector:result];
            status = finish(command, error);
        } @catch (NSException *exception) {
            set_error(error, [exception reason]);
        }
        [matrix release];
        [input release];
        [result release];
        [multiplication release];
        return status;
    }
}
