//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch
// and solving a block (#286819) which the information is downloaded from Block Explorer
//***********************************************************************************
// #define DEBUG

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "sha256.h"

#define numberOfSMs 20
#define numberOfWarps 32
//  256 160
#define threadsPerBlock 256
#define blocksPerGrid 160
#define totalThreads (threadsPerBlock * blocksPerGrid)

/*
Device 0: "GeForce GTX 1080"
  CUDA Driver Version / Runtime Version          11.0 / 11.0
  CUDA Capability Major/Minor version number:    6.1
  Total amount of global memory:                 8112 MBytes (8505524224 bytes)
  (20) Multiprocessors, (128) CUDA Cores/MP:     2560 CUDA Cores
  GPU Max Clock rate:                            1835 MHz (1.84 GHz)
  Memory Clock rate:                             5005 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 2097152 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
 */

#define _rotl(v, s) ((v) << (s) | (v) >> (32 - (s)))
#define _rotr(v, s) ((v) >> (s) | (v) << (32 - (s)))
#define _swap(x, y) (((x) ^= (y)), ((y) ^= (x)), ((x) ^= (y)))

#ifdef DEBUG
#include <chrono>
#define __debug_printf(fmt, args...) printf(fmt, ##args);
#define __START_TIME(ID) auto start_##ID = std::chrono::high_resolution_clock::now();
#define __END_TIME(ID)                                                                                        \
    auto stop_##ID = std::chrono::high_resolution_clock::now();                                               \
    int duration_##ID = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_##ID - start_##ID).count(); \
    __debug_printf("duration of %s: %d nanoseconds\n", #ID, duration_##ID);
#else
#define __debug_printf(fmt, args...)
#define __START_TIME(ID)
#define __END_TIME(ID)
#endif

__constant__ WORD k_gpu[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

////////////////////////   Block   /////////////////////

typedef struct _block {
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
} HashBlock;

////////////////////////   Utils   ///////////////////////

// convert one hex-codec char to binary
unsigned char decode(unsigned char c) {
    switch (c) {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
        default:
            return c - '0';
    }
}

// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char *out, char *in, size_t string_len) {
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len / 2 - 1;

    for (/* s, b */; s < string_len; s += 2, --b) {
        out[b] = (unsigned char)(decode(in[s]) << 4) + decode(in[s + 1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char *hex, size_t len) {
#ifdef DEBUG
    for (int i = 0; i < len; ++i) {
        __debug_printf("%02x", hex[i]);
    }
#endif
}

// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char *hex, size_t len) {
#ifdef DEBUG
    for (int i = len - 1; i >= 0; --i) {
        __debug_printf("%02x", hex[i]);
    }
#endif
}

__device__ void print_sha256_inverse_gpu(unsigned char *hex) {
#ifdef DEBUG
#pragma unroll
    for (int i = 31; i >= 0; --i) {
        __debug_printf("%02x", hex[i]);
    }
#endif
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len) {
    // compared from lowest bit
    for (int i = byte_len - 1; i >= 0; --i) {
        if (a[i] < b[i])
            return -1;
        else if (a[i] > b[i])
            return 1;
    }
    return 0;
}

__device__ int little_endian_bit_comparison_gpu(const unsigned char *a, const unsigned char *b) {
#pragma unroll
    // compared from lowest bit
    for (int i = 31; i >= 0; --i) {
        if (a[i] < b[i])
            return -1;
        else if (a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp) {
    int i = 0;
    while (i < len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n')
        ;
    str[len - 1] = '\0';
}

void cudaErrorPrint(cudaError_t err) {
#ifdef DEBUG
    if (err != cudaSuccess) {
        __debug_printf("Error: %s\n", cudaGetErrorString(err));
    }
#endif
}

void cudaErrorPrint(cudaError_t err, int idx) {
#ifdef DEBUG
    if (err != cudaSuccess) {
        __debug_printf("Error %d: %s\n", idx, cudaGetErrorString(err));
    }
#endif
}

////////////////////////   Hash   ///////////////////////

__device__ void sha256_transform_gpu(SHA256 *ctx, const BYTE *msg) {
    // Create a 64-entry message schedule array w[0..63] of 32-bit words
    WORD w[64];
#pragma unroll
    // Copy chunk into first 16 words w[0..15] of the message schedule array
    for (uint8_t i = 0, j = 0; i < 16; ++i, j += 4) {
        w[i] = (msg[j] << 24) | (msg[j + 1] << 16) | (msg[j + 2] << 8) | (msg[j + 3]);
    }

#pragma unroll
    // Extend the first 16 words into the remaining 48 words w[16..63] of the message schedule array:
    for (uint8_t i = 16; i < 64; ++i) {
        WORD s0 = (_rotr(w[i - 15], 7)) ^ (_rotr(w[i - 15], 18)) ^ (w[i - 15] >> 3);
        WORD s1 = (_rotr(w[i - 2], 17)) ^ (_rotr(w[i - 2], 19)) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    // Initialize working variables to current hash value
    WORD a = ctx->h[0];
    WORD b = ctx->h[1];
    WORD c = ctx->h[2];
    WORD d = ctx->h[3];
    WORD e = ctx->h[4];
    WORD f = ctx->h[5];
    WORD g = ctx->h[6];
    WORD h = ctx->h[7];

#pragma unroll
    // Compress function main loop:
    for (uint8_t i = 0; i < 64; ++i) {
        WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
        WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
        WORD ch = (e & f) ^ ((~e) & g);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp1 = h + S1 + ch + k_gpu[i] + w[i];
        WORD temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Add the compressed chunk to the current hash value
    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

__device__ void sha256_gpu(SHA256 *ctx, const BYTE *msg, size_t len) {
    // Initialize hash values:
    // (first 32 bits of the fractional parts of the square roots of the first 8 primes 2..19):
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;

    uint8_t i, j;
    size_t remain = len % 64;
    size_t total_len = len - remain;

    // Process the message in successive 512-bit chunks
    // For each chunk:
    for (i = 0; i < total_len; i += 64) {
        sha256_transform_gpu(ctx, &msg[i]);
    }

    // Process remain data
    BYTE m[64] = {};
    for (i = total_len, j = 0; i < len; ++i, ++j) {
        m[j] = msg[i];
    }

    // Append a single '1' bit
    m[j++] = 0x80;  // 1000 0000

    // Append K '0' bits, where k is the minimum number >= 0 such that L + 1 + K + 64 is a multiple of 512
    if (j > 56) {
        sha256_transform_gpu(ctx, m);
        memset(m, 0, sizeof(m));
        __debug_printf("true\n");
    }

    // Append L as a 64-bit bug-endian integer, making the total post-processed length a multiple of 512 bits
    unsigned long long L = len * 8;  // bits
    m[63] = L;
    m[62] = L >> 8;
    m[61] = L >> 16;
    m[60] = L >> 24;
    m[59] = L >> 32;
    m[58] = L >> 40;
    m[57] = L >> 48;
    m[56] = L >> 56;
    sha256_transform_gpu(ctx, m);

#pragma unroll
    // Produce the final hash value (little-endian to big-endian)
    // Swap 1st & 4th, 2nd & 3rd byte for each word
    for (i = 0; i < 32; i += 4) {
        _swap(ctx->b[i], ctx->b[i + 3]);
        _swap(ctx->b[i + 1], ctx->b[i + 2]);
    }
}

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len) {
    SHA256 tmp;
    sha256(&tmp, (BYTE *)bytes, len);
    sha256(sha256_ctx, (BYTE *)&tmp, sizeof(tmp));
}

__device__ void double_sha256_gpu(SHA256 *sha256_ctx, unsigned char *bytes) {
    SHA256 tmp;

    // sha256_gpu(&tmp, (BYTE *)bytes, 80);
    {
        (&tmp)->h[0] = 0x6a09e667;
        (&tmp)->h[1] = 0xbb67ae85;
        (&tmp)->h[2] = 0x3c6ef372;
        (&tmp)->h[3] = 0xa54ff53a;
        (&tmp)->h[4] = 0x510e527f;
        (&tmp)->h[5] = 0x9b05688c;
        (&tmp)->h[6] = 0x1f83d9ab;
        (&tmp)->h[7] = 0x5be0cd19;

        sha256_transform_gpu((&tmp), &bytes[0]);

        BYTE m[64] = {};
#pragma unroll
        for (uint8_t j = 0; j < 16; ++j) {
            m[j] = bytes[j + 64];
        }

        m[63] = 128U;
        m[62] = 2U;
        m[16] = 0x80U;
        sha256_transform_gpu((&tmp), m);

#pragma unroll
        for (uint8_t i = 0; i < 32; i += 4) {
            _swap((&tmp)->b[i], (&tmp)->b[i + 3]);
            _swap((&tmp)->b[i + 1], (&tmp)->b[i + 2]);
        }
    }

    // sha256_gpu(sha256_ctx, (BYTE *)&tmp, 32);
    {
        sha256_ctx->h[0] = 0x6a09e667;
        sha256_ctx->h[1] = 0xbb67ae85;
        sha256_ctx->h[2] = 0x3c6ef372;
        sha256_ctx->h[3] = 0xa54ff53a;
        sha256_ctx->h[4] = 0x510e527f;
        sha256_ctx->h[5] = 0x9b05688c;
        sha256_ctx->h[6] = 0x1f83d9ab;
        sha256_ctx->h[7] = 0x5be0cd19;

        // Process remain data
        BYTE m[64] = {};
#pragma unroll
        for (uint8_t i = 0; i < 32; ++i) {
            m[i] = ((BYTE *)&tmp)[i];
        }

        m[62] = 1U;
        m[32] = 0x80U;  // 1000 0000
        sha256_transform_gpu(sha256_ctx, m);

#pragma unroll
        for (uint8_t i = 0; i < 32; i += 4) {
            _swap(sha256_ctx->b[i], sha256_ctx->b[i + 3]);
            _swap(sha256_ctx->b[i + 1], sha256_ctx->b[i + 2]);
        }
    }
}

////////////////////   Merkle Root   /////////////////////

// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch) {
    size_t total_count = count;  // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count + 1) * 32];
    unsigned char **list = new unsigned char *[total_count + 1];

    // copy each branch to the list
    for (int i = 0; i < total_count; ++i) {
        list[i] = raw_list + i * 32;
        // convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count * 32;

    // calculate merkle root
    while (total_count > 1) {
        // hash each pair
        int i, j;

        if (total_count % 2 == 1)  // odd,
        {
            memcpy(list[total_count], list[total_count - 1], 32);
        }

        for (i = 0, j = 0; i < total_count; i += 2, ++j) {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256 *)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

//////////////////////   SHA256   ////////////////////////

__global__ void get_nonce(unsigned char target_hex[], const unsigned int version, const unsigned char prevhash[], const unsigned char merkle_root[], const unsigned int ntime, const unsigned int nbits, unsigned int *nonce, int *found_cnt) {
    if (*found_cnt != 0)
        return;

    int index = blockIdx.x * threadsPerBlock + threadIdx.x;
    const unsigned int thride = totalThreads;
    // __debug_printf("index %d start\n", index);

    // __shared__ unsigned char local_target_hex[32];
    // __shared__ unsigned int local_version;
    // __shared__ unsigned char local_prevhash[32];
    // __shared__ unsigned char local_merkle_root[32];
    // __shared__ unsigned int local_ntime;
    // __shared__ unsigned int local_nbits;

    // if (threadIdx.x < 32) {
    //     local_target_hex[threadIdx.x] = target_hex[threadIdx.x];
    // } else if (threadIdx.x < 64) {
    //     local_prevhash[threadIdx.x - 32] = prevhash[threadIdx.x - 32];
    // } else if (threadIdx.x < 96) {
    //     local_merkle_root[threadIdx.x - 64] = merkle_root[threadIdx.x - 64];
    // } else if (threadIdx.x < 97) {
    //     local_version = version;
    // } else if (threadIdx.x < 98) {
    //     local_ntime = ntime;
    // } else if (threadIdx.x < 99) {
    //     local_nbits = nbits;
    // }
    // __syncthreads();

    SHA256 sha256_ctx;
    HashBlock block;

    block.nonce = index;

    // #pragma unroll
    //     for (int8_t i = 0; i < 32; i++)
    //         block.prevhash[i] = local_prevhash[i];
    // #pragma unroll
    //     for (int8_t i = 0; i < 32; i++)
    //         block.merkle_root[i] = local_merkle_root[i];
    block.version = version;
    // memcpy(block.prevhash, prevhash, 32);
    // memcpy(block.merkle_root, merkle_root, 32);
#pragma unroll
    for (int8_t i = 0; i < 32; i++)
        block.prevhash[i] = prevhash[i];
#pragma unroll
    for (int8_t i = 0; i < 32; i++)
        block.merkle_root[i] = merkle_root[i];
    block.ntime = ntime;
    block.nbits = nbits;

    while (true) {
        // sha256d
        double_sha256_gpu(&sha256_ctx, (unsigned char *)&block);
#ifdef DEBUG
        if (block.nonce % 100000000 == 0) {
            __debug_printf("index %5u ", index);
            __debug_printf("hash #%10u (big): ", block.nonce);
            print_sha256_inverse_gpu(sha256_ctx.b);
            __debug_printf("\n");
        }
#endif

        if (little_endian_bit_comparison_gpu(sha256_ctx.b, /* local_ */ target_hex) < 0)  // sha256_ctx < target_hex
        {
#ifdef DEBUG
            __debug_printf("Found Solution!!\n");
            __debug_printf("index %5u ", index);
            __debug_printf("hash #%10u (big): ", block.nonce);
            print_sha256_inverse_gpu(sha256_ctx.b);
            __debug_printf("\n\n");
#endif
            if (0 == atomicAdd(found_cnt, 1))
                *nonce = block.nonce;
        }
        // __syncthreads();

        if (*found_cnt != 0 || 0xffffffff - thride < block.nonce)
            break;
        else
            block.nonce += thride;
    }
}

void solve(FILE *fin, FILE *fout) {
    // **** read data *****
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);
    __debug_printf("start hashing");

    raw_merkle_branch = new char[tx * 65];
    merkle_branch = new char *[tx];
    for (int i = 0; i < tx; ++i) {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // **** calculate merkle root ****

    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    __debug_printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    __debug_printf("\n");

    __debug_printf("merkle root(big):    ");
    print_hex_inverse(merkle_root, 32);
    __debug_printf("\n");

    // **** solve block ****
    __debug_printf("Block info (big): \n");
    __debug_printf("  version:  %s\n", version);
    __debug_printf("  pervhash: %s\n", prevhash);
    __debug_printf("  merkleroot: ");
    print_hex_inverse(merkle_root, 32);
    __debug_printf("\n");
    __debug_printf("  nbits:    %s\n", nbits);
    __debug_printf("  ntime:    %s\n", ntime);
    __debug_printf("  nonce:    ???\n\n");

    HashBlock block;

    // convert to byte array in little-endian
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;

    // ********** calculate target value *********
    // calculate target value from encoded difficulty which is encoded on "nbits"
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};

    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;

    // little-endian
    target_hex[sb] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8 - rb));
    target_hex[sb + 2] = (mant >> (16 - rb));
    target_hex[sb + 3] = (mant >> (24 - rb));

    __debug_printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    __debug_printf("\n");

    // ********** find nonce **************

    SHA256 sha256_ctx;

    int *found_cnt;
    unsigned char *device_prevhash;
    unsigned char *device_merkle_root;
    unsigned char *device_target_hex;
    unsigned int *device_nonce;
    cudaErrorPrint(cudaMalloc(&found_cnt, 4), 0);
    cudaErrorPrint(cudaMalloc(&device_prevhash, 32), 1);     // Cuda malloc gpu previous hash
    cudaErrorPrint(cudaMalloc(&device_merkle_root, 32), 2);  // Cuda malloc gpu merkle root
    cudaErrorPrint(cudaMalloc(&device_target_hex, 32), 3);   // Cuda malloc gpu target hex
    cudaErrorPrint(cudaMalloc(&device_nonce, 4), 4);         // Cuda malloc gpu nonce array
    cudaErrorPrint(cudaMemset(found_cnt, 0, (size_t)4), 40);
    cudaErrorPrint(cudaMemcpyAsync(device_prevhash, block.prevhash, 32, cudaMemcpyHostToDevice), 5);        // Cuda asyncronize copy previous hash
    cudaErrorPrint(cudaMemcpyAsync(device_merkle_root, block.merkle_root, 32, cudaMemcpyHostToDevice), 6);  // Cuda asyncronize copy merkle root
    cudaErrorPrint(cudaMemcpyAsync(device_target_hex, target_hex, 32, cudaMemcpyHostToDevice), 7);          // Cuda asyncronize copy target hex

    dim3 myBlockDim(threadsPerBlock);
    dim3 myGridDim(blocksPerGrid);
    get_nonce<<<myGridDim, myBlockDim>>>(device_target_hex, block.version, device_prevhash, device_merkle_root, block.ntime, block.nbits, device_nonce, found_cnt);
    cudaErrorPrint(cudaMemcpyAsync(&block.nonce, device_nonce, 4, cudaMemcpyDeviceToHost), 8);  // Cuda asyncronize copy nonce

    // print result

    // little-endian
    __debug_printf("hash(little): ");
    print_hex(sha256_ctx.b, 32);
    __debug_printf("\n");

    // big-endian
    __debug_printf("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    __debug_printf("\n\n");

    cudaErrorPrint(cudaGetLastError(), 9);        // Cuda debug
    cudaErrorPrint(cudaDeviceSynchronize(), 10);  // Cuda synchronize
    for (int i = 0; i < 4; ++i) {
        fprintf(fout, "%02x", ((unsigned char *)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    delete[] merkle_branch;
    delete[] raw_merkle_branch;
    cudaFree(found_cnt);
    cudaFree(device_prevhash);
    cudaFree(device_merkle_root);
    cudaFree(device_target_hex);
    cudaFree(device_nonce);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
    }
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for (int i = 0; i < totalblock; ++i) {
        solve(fin, fout);
    }

    return 0;
}

/*
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 ./hw4 testcases/case00.in outputs/case00.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 ./hw4 testcases/case01.in outputs/case01.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 ./hw4 testcases/case02.in outputs/case02.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 ./hw4 testcases/case03.in outputs/case03.out;

make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 cuda-memcheck ./hw4 testcases/case00.in outputs/case00.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 cuda-memcheck ./hw4 testcases/case01.in outputs/case01.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 cuda-memcheck ./hw4 testcases/case02.in outputs/case02.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 cuda-memcheck ./hw4 testcases/case03.in outputs/case03.out;

make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 nvprof ./hw4 testcases/case00.in outputs/case00.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 nvprof ./hw4 testcases/case01.in outputs/case01.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 nvprof ./hw4 testcases/case02.in outputs/case02.out;
make; srun -p ipc22 -N1 -n1 -c1 -t1 --gres=gpu:1 nvprof ./hw4 testcases/case03.in outputs/case03.out;

./validation outputs/case00.out testcases/case00.out;
./validation outputs/case01.out testcases/case01.out;
./validation outputs/case02.out testcases/case02.out;
./validation outputs/case03.out testcases/case03.out;
 */