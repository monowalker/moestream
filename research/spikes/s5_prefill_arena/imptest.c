// Can RADV import a file-backed mmap? Compared against anonymous memory.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev;
static PFN_vkGetMemoryHostPointerPropertiesEXT pfnProps;
static int try_import(void *p, size_t sz, const char *label) {
    VkMemoryHostPointerPropertiesEXT hp = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
    VkResult r = pfnProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, p, &hp);
    if (r != VK_SUCCESS) { printf("  %-22s getProps failed (%d)\n", label, r); return 0; }
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(hp.memoryTypeBits & (1u << i))) continue;
        VkImportMemoryHostPointerInfoEXT imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
        imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = p;
        VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, sz, i };
        VkDeviceMemory m;
        VkResult ar = vkAllocateMemory(dev, &ai, NULL, &m);
        if (ar == VK_SUCCESS) { printf("  %-22s OK (memtype %u)\n", label, i); vkFreeMemory(dev,m,NULL); return 1; }
        printf("  %-22s memtype %u -> VkResult %d\n", label, i, ar);
    }
    printf("  %-22s failed on every memtype (bits=0x%x)\n", label, hp.memoryTypeBits);
    return 0;
}
int main(int argc, char**argv) {
    const char *exts[] = { 0 };
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO }; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ic = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO }; ic.pApplicationInfo = &app;
    if (vkCreateInstance(&ic, NULL, &inst) != VK_SUCCESS) { puts("instance creation failed"); return 1; }
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);
    float pri = 1.0f;
    VkDeviceQueueCreateInfo q = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO }; q.queueCount = 1; q.pQueuePriorities = &pri;
    const char *de[] = { "VK_EXT_external_memory_host" };
    VkDeviceCreateInfo dc = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dc.queueCreateInfoCount = 1; dc.pQueueCreateInfos = &q;
    dc.enabledExtensionCount = 1; dc.ppEnabledExtensionNames = de;
    if (vkCreateDevice(pd, &dc, NULL, &dev) != VK_SUCCESS) { puts("device creation failed"); return 1; }
    pfnProps = (PFN_vkGetMemoryHostPointerPropertiesEXT) vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    if (!pfnProps) { puts("extension function unavailable"); return 1; }
    const size_t SZ = 64u<<20;
    void *anon = mmap(NULL, SZ, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    memset(anon, 0, SZ);
    try_import(anon, SZ, "anonymous mmap (RW)");
    int fd = open(argv[1], O_RDONLY);
    void *fmap = mmap(NULL, SZ, PROT_READ, MAP_SHARED, fd, 0);
    if (fmap == MAP_FAILED) { puts("file mmap failed"); return 1; }
    ((volatile char*)fmap)[0];
    try_import(fmap, SZ, "file mmap (SHARED)");
    void *fpriv = mmap(NULL, SZ, PROT_READ, MAP_PRIVATE, fd, 0);
    ((volatile char*)fpriv)[0];
    try_import(fpriv, SZ, "file mmap (PRIVATE)");
    (void)exts; return 0;
}
