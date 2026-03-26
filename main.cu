#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define WIDTH 512
#define HEIGHT 512

// ================= GPU BLUR =================
__global__ void blurKernel(unsigned char* input, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int sum = 0;
    int count = 0;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && ny >= 0 && nx < WIDTH && ny < HEIGHT) {
                sum += input[ny * WIDTH + nx];
                count++;
            }
        }
    }

    output[y * WIDTH + x] = sum / count;
}

// ================= GPU EDGE =================
__global__ void edgeKernel(unsigned char* input, unsigned char* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;

    int idx = y * WIDTH + x;

    int gx = 0;
    int gy = 0;

    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1) {
        gx = -input[idx - 1] + input[idx + 1];
        gy = -input[idx - WIDTH] + input[idx + WIDTH];
    }

    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;

    output[idx] = mag;
}

// ================= MAIN =================
int main() {
    int size = WIDTH * HEIGHT * sizeof(unsigned char);

    unsigned char *h_input, *h_blur, *h_edge;
    h_input = (unsigned char*)malloc(size);
    h_blur = (unsigned char*)malloc(size);
    h_edge = (unsigned char*)malloc(size);

    // Generate dummy image
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_input[i] = rand() % 256;
    }

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Blur
    blurKernel<<<grid, block>>>(d_input, d_output);
    cudaMemcpy(h_blur, d_output, size, cudaMemcpyDeviceToHost);

    // Edge
    edgeKernel<<<grid, block>>>(d_input, d_output);
    cudaMemcpy(h_edge, d_output, size, cudaMemcpyDeviceToHost);

    printf("Processing complete on GPU!\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_blur);
    free(h_edge);

    return 0;
}
