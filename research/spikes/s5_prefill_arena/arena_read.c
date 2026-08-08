// Spike S5b — GPU read bandwidth of an imported anonymous arena
//
// S5a showed the import itself succeeds, but the memory obtained was
//   memtype=5 heap=0 [HOST_VISIBLE HOST_COHERENT HOST_CACHED] -- not DEVICE_LOCAL.
// The current slab uses heap 1, DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT.
// They are different, so measure and compare GPU-side read speed.
//
// Three conditions, read by the same compute shader, reported in GB/s:
//   A: the imported anonymous arena  (the path this proposal would use)
//   B: DEVICE_LOCAL only             (the pure device-memory ceiling)
//   C: DEVICE_LOCAL|HOST_VISIBLE     (the current MoEStream slab: the baseline)
//
// The A/C ratio decides whether the proposal lives or dies.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define EXPERT_BYTES 450560ull
#define N_EXPERT     256ull
#define LAYER_BYTES  (EXPERT_BYTES * N_EXPERT * 3ull)   // 330 MiB
#define WGS          256u
#define NWG          1024u
#define DST_BYTES    ((size_t) WGS * NWG * 16u)

static VkInstance       inst;
static VkPhysicalDevice pd;
static VkDevice         dev;
static VkQueue          queue;
static uint32_t         qfam;
static VkPhysicalDeviceMemoryProperties mp;
static PFN_vkGetMemoryHostPointerPropertiesEXT pfnProps;

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static void flags_str(VkMemoryPropertyFlags f, char *o, size_t n) {
    o[0] = 0;
    if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)  strncat(o, "DEVICE_LOCAL ",  n - strlen(o) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)  strncat(o, "HOST_VISIBLE ",  n - strlen(o) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) strncat(o, "HOST_COHERENT ", n - strlen(o) - 1);
    if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)   strncat(o, "HOST_CACHED ",   n - strlen(o) - 1);
}

// Find a memory type containing every req bit and none of the excl bits
static int find_type(uint32_t bits, VkMemoryPropertyFlags req, VkMemoryPropertyFlags excl) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & req) != req)  continue;
        if (f & excl)          continue;
        return (int) i;
    }
    return -1;
}

struct buf { VkBuffer b; VkDeviceMemory m; int type; };

// Import a host pointer when import_ptr is non-NULL
static int make_buf(struct buf *out, size_t sz, VkMemoryPropertyFlags req,
                    VkMemoryPropertyFlags excl, void *import_ptr, const char *label) {
    memset(out, 0, sizeof *out);
    VkExternalMemoryBufferCreateInfo ext = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    if (import_ptr) bci.pNext = &ext;
    bci.size  = sz;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &bci, NULL, &out->b) != VK_SUCCESS) return 0;

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, out->b, &mr);

    uint32_t bits = mr.memoryTypeBits;
    VkImportMemoryHostPointerInfoEXT imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
    if (import_ptr) {
        VkMemoryHostPointerPropertiesEXT hp = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        if (pfnProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                     import_ptr, &hp) != VK_SUCCESS) { printf("  %s: getProps failed\n", label); return 0; }
        bits &= hp.memoryTypeBits;
        imp.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        imp.pHostPointer = import_ptr;
    }
    int t = find_type(bits, req, excl);
    if (t < 0) { printf("  %s: no matching memory type\n", label); return 0; }

    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.pNext = import_ptr ? (void *) &imp : NULL;
    ai.allocationSize  = import_ptr ? sz : mr.size;
    ai.memoryTypeIndex = (uint32_t) t;
    VkResult r = vkAllocateMemory(dev, &ai, NULL, &out->m);
    if (r != VK_SUCCESS) { printf("  %s: vkAllocateMemory failed (VkResult %d)\n", label, r); return 0; }
    vkBindBufferMemory(dev, out->b, out->m, 0);
    out->type = t;

    char fs[128]; flags_str(mp.memoryTypes[t].propertyFlags, fs, sizeof fs);
    printf("  %-42s memtype=%u heap=%u [%s]\n", label, t, mp.memoryTypes[t].heapIndex, fs);
    return 1;
}

int main(void) {
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ic = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ic.pApplicationInfo = &app;
    if (vkCreateInstance(&ic, NULL, &inst) != VK_SUCCESS) { puts("instance creation failed"); return 1; }
    uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
    if (qfam == UINT32_MAX) { puts("no compute queue"); return 1; }

    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &pri;
    const char *de[] = { "VK_EXT_external_memory_host" };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = de;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { puts("device creation failed"); return 1; }
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    pfnProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)
        vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);

    // ---- shader ----
    FILE *f = fopen("/tmp/read.spv", "rb");
    if (!f) { puts("/tmp/read.spv not found"); return 1; }
    fseek(f, 0, SEEK_END); long slen = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(slen);
    if (fread(code, 1, slen, f) != (size_t) slen) { puts("failed to read the spv"); return 1; }
    fclose(f);
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = slen; smci.pCode = code;
    VkShaderModule sm;
    if (vkCreateShaderModule(dev, &smci, NULL, &sm) != VK_SUCCESS) { puts("shader module creation failed"); return 1; }

    VkDescriptorSetLayoutBinding lb[2] = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
    };
    VkDescriptorSetLayoutCreateInfo dlci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dlci.bindingCount = 2; dlci.pBindings = lb;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl);

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &plci, NULL, &pl);

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe;
    if (vkCreateComputePipelines(dev, NULL, 1, &cpci, NULL, &pipe) != VK_SUCCESS) { puts("pipeline creation failed"); return 1; }

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = 4; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, NULL, &dp);

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.queueFamilyIndex = qfam;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; vkCreateCommandPool(dev, &cpi, NULL, &cp);
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(dev, &cbai, &cb);
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence; vkCreateFence(dev, &fci, NULL, &fence);

    // ---- buffers ----
    printf("=== buffer allocation ===\n");
    struct buf dst;
    if (!make_buf(&dst, DST_BYTES, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, NULL, "dst (DEVICE_LOCAL)")) return 1;

    void *anon = mmap(NULL, LAYER_BYTES, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    memset(anon, 0x5a, LAYER_BYTES);

    struct buf A, B, C;
    int hasA = make_buf(&A, LAYER_BYTES, 0, 0, anon, "A: imported anonymous arena");
    int hasB = make_buf(&B, LAYER_BYTES, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, NULL, "B: DEVICE_LOCAL only");
    int hasC = make_buf(&C, LAYER_BYTES,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        0, NULL, "C: DEVICE_LOCAL|HOST_VISIBLE (the current slab)");

    const uint32_t nvec = (uint32_t) (LAYER_BYTES / 16);
    printf("\n=== compute-shader reads (%.0f MiB per run, best of 5) ===\n",
           LAYER_BYTES / 1048576.0);

    struct { struct buf *b; int ok; const char *label; } runs[] = {
        { &A, hasA, "A: imported anonymous arena " },
        { &B, hasB, "B: DEVICE_LOCAL only        " },
        { &C, hasC, "C: DEVICE_LOCAL|HOST_VISIBLE" },
    };
    double gbps[3] = {0,0,0};

    for (int r = 0; r < 3; ++r) {
        if (!runs[r].ok) { printf("  %s  — allocation failed\n", runs[r].label); continue; }
        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
        VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsai, &ds);
        VkDescriptorBufferInfo bi[2] = {
            { runs[r].b->b, 0, VK_WHOLE_SIZE }, { dst.b, 0, VK_WHOLE_SIZE } };
        VkWriteDescriptorSet w[2] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[0], NULL },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &bi[1], NULL } };
        vkUpdateDescriptorSets(dev, 2, w, 0, NULL);

        double best = 0;
        for (int it = 0; it < 6; ++it) {          // the first iteration is a warm-up
            VkCommandBufferBeginInfo bbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkResetCommandBuffer(cb, 0);
            vkBeginCommandBuffer(cb, &bbi);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &nvec);
            vkCmdDispatch(cb, NWG, 1, 1);
            vkEndCommandBuffer(cb);

            VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            vkResetFences(dev, 1, &fence);
            double t0 = now_s();
            vkQueueSubmit(queue, 1, &si, fence);
            vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
            double dt = now_s() - t0;
            if (it == 0) continue;
            double g = LAYER_BYTES / dt / 1e9;
            if (g > best) best = g;
        }
        gbps[r] = best;
        printf("  %s  %7.2f GB/s\n", runs[r].label, best);
    }

    if (gbps[0] > 0 && gbps[2] > 0)
        printf("\n  A / C = %.2f  (imported arena relative to the current slab path)\n", gbps[0] / gbps[2]);
    if (gbps[0] > 0 && gbps[1] > 0)
        printf("  A / B = %.2f  (relative to pure device memory)\n", gbps[0] / gbps[1]);
    return 0;
}
