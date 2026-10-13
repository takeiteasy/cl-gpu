// C ABI over Metal for cl-gpu. No ARC: constructors return +1 objects and clgpu_release frees them.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
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
