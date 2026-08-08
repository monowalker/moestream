// Spike S5 — feasibility of an anonymous staging arena for prefill
//
// What this checks (the proposal dies unless these hold):
//   (1) can one layer's full expert set (330 MiB) be imported into Vulkan as
//       anonymous memory?
//   (2) does it still work as the size scales (660 MiB double-buffered,
//       1320 MiB for four layers)?
//   (3) what properties does the imported memory have -- is it DEVICE_LOCAL,
//       i.e. readable directly by the GPU?
//   (4) what is the effective bandwidth reading into it from SSD with O_DIRECT?
//       (O_DIRECT is mandatory so the page cache's apparent bandwidth is not
//        mistaken for the real thing)
//
// As a control, importing a file-backed mmap is attempted at the same time,
// confirming that RESULTS.md §11's "RADV rejects with VkResult -13" reproduces.
//
//   Usage: ./arena_import <path to a gguf>
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

#define EXPERT_BYTES 450560ull            // from the on-device diagnostics
#define N_EXPERT     256ull
#define LAYER_BYTES  (EXPERT_BYTES * N_EXPERT * 3ull)   // gate/up/down = 330 MiB

static VkInstance       inst;
static VkPhysicalDevice pd;
static VkDevice         dev;
static PFN_vkGetMemoryHostPointerPropertiesEXT pfnProps;
static VkPhysicalDeviceMemoryProperties mp;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void flags_str(VkMemoryPropertyFlags f, char *out, size_t n) {
    out[0] = 0;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)     strncat(out, "DEVICE_LOCAL ",  n - strlen(out) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)     strncat(out, "HOST_VISIBLE ",  n - strlen(out) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)    strncat(out, "HOST_COHERENT ", n - strlen(out) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)      strncat(out, "HOST_CACHED ",   n - strlen(out) - 1);
}

// Attempt the import. On success the memory is returned via *out_mem and the
// caller is responsible for freeing it.
static int try_import(void *p, size_t sz, const char *label, VkDeviceMemory *out_mem) {
    VkMemoryHostPointerPropertiesEXT hp = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
    VkResult r = pfnProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, p, &hp);
    if (r != VK_SUCCESS) {
        printf("  %-28s getMemoryHostPointerProperties failed (VkResult %d)\n", label, r);
        return 0;
    }
    VkResult last = VK_SUCCESS;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(hp.memoryTypeBits & (1u << i))) continue;
        VkImportMemoryHostPointerInfoEXT imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
        imp.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = p;
        VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, sz, i };
        VkDeviceMemory m;
        VkResult ar = vkAllocateMemory(dev, &ai, NULL, &m);
        if (ar == VK_SUCCESS) {
            char fs[128];
            flags_str(mp.memoryTypes[i].propertyFlags, fs, sizeof fs);
            printf("  %-28s OK  memtype=%u heap=%u  [%s]\n",
                   label, i, mp.memoryTypes[i].heapIndex, fs);
            if (out_mem) *out_mem = m; else vkFreeMemory(dev, m, NULL);
            return 1;
        }
        last = ar;
    }
    printf("  %-28s failed (VkResult %d, memoryTypeBits=0x%x)\n", label, last, hp.memoryTypeBits);
    return 0;
}

// ---- parallel O_DIRECT pread ------------------------------------------------
struct job { int fd; char *dst; off_t off; size_t len; };
static void * worker(void *a) {
    struct job *j = (struct job *) a;
    size_t done = 0;
    while (done < j->len) {
        ssize_t n = pread(j->fd, j->dst + done, j->len - done, j->off + done);
        if (n <= 0) { if (n < 0) fprintf(stderr, "  pread: %s\n", strerror(errno)); break; }
        done += (size_t) n;
    }
    j->len = done;   // report how much was actually read
    return NULL;
}

static void bandwidth_test(const char *path, char *arena, size_t sz, int nthreads) {
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { printf("  O_DIRECT open failed: %s\n", strerror(errno)); return; }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t) st.st_size < sz * 2) {
        printf("  file is too small\n"); close(fd); return;
    }
    // Read from a 4 KiB-aligned offset near the end, unlikely to be cached
    off_t base = (off_t) (((size_t) st.st_size - sz) & ~4095ull);

    pthread_t th[64]; struct job jb[64];
    if (nthreads > 64) nthreads = 64;
    size_t chunk = (sz / nthreads) & ~4095ull;

    double t0 = now_s();
    for (int i = 0; i < nthreads; ++i) {
        jb[i].fd  = fd;
        jb[i].dst = arena + (size_t) i * chunk;
        jb[i].off = base + (off_t) ((size_t) i * chunk);
        jb[i].len = (i == nthreads - 1) ? (sz - (size_t) i * chunk) & ~4095ull : chunk;
        pthread_create(&th[i], NULL, worker, &jb[i]);
    }
    size_t total = 0;
    for (int i = 0; i < nthreads; ++i) { pthread_join(th[i], NULL); total += jb[i].len; }
    double dt = now_s() - t0;
    close(fd);

    printf("  O_DIRECT pread x%d : %.1f MiB in %.3f s -> %.2f GB/s\n",
           nthreads, total / 1048576.0, dt, total / dt / 1e9);
    if (dt > 0) {
        double full = LAYER_BYTES * 40.0;    // 40 layers = every expert in the model
        printf("     -> estimated time for one forward pass over all 40 layers (%.2f GiB): %.2f s\n",
               full / 1073741824.0, full / (total / dt));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { puts("usage: arena_import <gguf>"); return 1; }

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ic = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ic.pApplicationInfo = &app;
    if (vkCreateInstance(&ic, NULL, &inst) != VK_SUCCESS) { puts("instance creation failed"); return 1; }
    uint32_t n = 1;
    vkEnumeratePhysicalDevices(inst, &n, &pd);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT ehp = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT };
    VkPhysicalDeviceMaintenance3Properties m3 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, &ehp };
    VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m3 };
    vkGetPhysicalDeviceProperties2(pd, &p2);

    printf("=== device ===\n");
    printf("  %s\n", props.deviceName);
    printf("  minImportedHostPointerAlignment = %llu\n",
           (unsigned long long) ehp.minImportedHostPointerAlignment);
    printf("  maxMemoryAllocationSize         = %.2f GiB\n",
           m3.maxMemoryAllocationSize / 1073741824.0);

    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("\n=== memory heaps ===\n");
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i)
        printf("  heap %u: %6.2f GiB %s\n", i, mp.memoryHeaps[i].size / 1073741824.0,
               (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL" : "");

    float pri = 1.0f;
    VkDeviceQueueCreateInfo q = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    q.queueCount = 1; q.pQueuePriorities = &pri;
    const char *de[] = { "VK_EXT_external_memory_host" };
    VkDeviceCreateInfo dc = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &q;
    dc.enabledExtensionCount = 1; dc.ppEnabledExtensionNames = de;
    if (vkCreateDevice(pd, &dc, NULL, &dev) != VK_SUCCESS) { puts("device creation failed"); return 1; }
    pfnProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!pfnProps) { puts("extension function unavailable"); return 1; }

    // ---- (1)(2) size scaling ----
    printf("\n=== importing an anonymous arena (size scaling) ===\n");
    struct { size_t sz; const char *label; } cases[] = {
        { 64ull << 20,       "64 MiB (control)" },
        { LAYER_BYTES,       "330 MiB = one layer" },
        { LAYER_BYTES * 2,   "660 MiB = double-buffered" },
        { LAYER_BYTES * 4,   "1320 MiB = four layers" },
    };
    void  *keep_ptr = NULL; size_t keep_sz = 0;
    for (unsigned i = 0; i < sizeof cases / sizeof cases[0]; ++i) {
        void *a = mmap(NULL, cases[i].sz, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (a == MAP_FAILED) { printf("  %-28s mmap failed\n", cases[i].label); continue; }
        memset(a, 0, cases[i].sz);   // force the pages to be backed
        VkDeviceMemory m = VK_NULL_HANDLE;
        int ok = try_import(a, cases[i].sz, cases[i].label, &m);
        if (ok && cases[i].sz == LAYER_BYTES) {
            keep_ptr = a; keep_sz = cases[i].sz;   // reuse for the bandwidth test
            vkFreeMemory(dev, m, NULL);
            continue;
        }
        if (m) vkFreeMemory(dev, m, NULL);
        munmap(a, cases[i].sz);
    }

    // ---- control: file-backed mmap ----
    printf("\n=== control: importing a file-backed mmap ===\n");
    int fd = open(argv[1], O_RDONLY);
    if (fd >= 0) {
        void *fs = mmap(NULL, LAYER_BYTES, PROT_READ, MAP_SHARED,  fd, 0);
        if (fs != MAP_FAILED) { ((volatile char*)fs)[0]; try_import(fs, LAYER_BYTES, "330 MiB SHARED", NULL);  munmap(fs, LAYER_BYTES); }
        void *fp = mmap(NULL, LAYER_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
        if (fp != MAP_FAILED) { ((volatile char*)fp)[0]; try_import(fp, LAYER_BYTES, "330 MiB PRIVATE", NULL); munmap(fp, LAYER_BYTES); }
        close(fd);
    }

    // ---- (4) effective bandwidth ----
    if (keep_ptr) {
        printf("\n=== O_DIRECT reads into the imported arena ===\n");
        for (int t = 1; t <= 16; t *= 2) bandwidth_test(argv[1], (char *) keep_ptr, keep_sz, t);
        munmap(keep_ptr, keep_sz);
    }
    return 0;
}
