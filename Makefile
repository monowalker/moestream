# MoEStream — every build runs inside Docker; nothing is installed on the host.
#
#   The product is run by docker compose. This Makefile is for measurement and
#   verification only.
#
#     product      docker compose up -d          (Dockerfile + .env)
#     measurement  make bench / make stats       (research/tools/)
#     verification make spike SPIKE=s0_slot_slab (research/spikes/, never shipped)
#
IMAGE_PREFIX ?= moestream
IMAGE_DEV    := $(IMAGE_PREFIX)-dev:local
IMAGE_DEV_VK := $(IMAGE_PREFIX)-dev-vk:local
IMAGE_SPIKE  := $(IMAGE_PREFIX)-spike:local
IMAGE_LLAMA  := $(IMAGE_PREFIX)-llama:local
IMAGE_TOOL   := $(IMAGE_PREFIX)-tool:local
SPIKE        ?= s0_slot_slab
TOOL         ?= expert_trace

.PHONY: up down logs test bench stats dev dev-vk spike spike-vk tool clean-docker images

# =============================================================================
# Product — thin wrappers around compose.yaml
# =============================================================================
## Start the server. The render/video GIDs are host facts, so they are detected
## here rather than being written into .env -- a stale GID there means no GPU,
## and the entrypoint can only report that after the fact, not prevent it.
up:
	@RENDER_GID=$$(getent group render | cut -d: -f3) \
	 VIDEO_GID=$$(getent group video  | cut -d: -f3) \
	 sh -c 'echo "[make] detected RENDER_GID=$$RENDER_GID VIDEO_GID=$$VIDEO_GID"; \
	        RENDER_GID=$$RENDER_GID VIDEO_GID=$$VIDEO_GID docker compose up -d'

down:
	docker compose down

logs:
	docker compose logs -f

# =============================================================================
# Tests — no GPU, no model, no Docker; seconds to run
# =============================================================================
## Run everything that can be checked without hardware.
##   Catches obvious regressions, not performance or output correctness --
##   those need the real machine (make bench). See research/tests/README.md.
test:
	@research/tests/run_all.sh

# =============================================================================
# Measurement (research/tools/)
# =============================================================================
## Measure speed, reporting the environment, slot count and output correctness
## alongside it (docs/RESULTS.md §12.1)
##   e.g. make bench ARGS="--baseline --frac 0.25"
##   Runs under research/tools/exclusive.sh: two measurements at once fight over the GPU
##   and over ./state, and produce numbers that look plausible but are wrong.
bench:
	@research/tools/exclusive.sh research/tools/ms-bench.sh $(ARGS)

## Pull statistics out of a running server (hit rate, recommended frac / UBATCH)
stats:
	@research/tools/ms-stats.sh $(or $(C),moestream)

# =============================================================================
# Verification (research/spikes/ and research/tools/analysis/) — never part of the product image
# =============================================================================
## Development image, with ggml prebuilt at a pinned commit
dev:
	docker build -f research/docker/Dockerfile.dev -t $(IMAGE_DEV) research/docker/

## Development image with Vulkan
dev-vk:
	docker build -f research/docker/Dockerfile.dev --build-arg WITH_VULKAN=ON -t $(IMAGE_DEV_VK) research/docker/

## Build and run a spike, e.g. make spike SPIKE=s0_slot_slab
spike: dev
	docker build -f research/docker/Dockerfile.spike \
	  --build-arg BASE=$(IMAGE_DEV) --build-arg SPIKE=$(SPIKE) \
	  -t $(IMAGE_SPIKE) research/
	docker run --rm $(IMAGE_SPIKE)

## Vulkan spike, run with the iGPU passed through
## e.g. make spike-vk SPIKE=s0b_backend_slab
spike-vk: dev-vk
	docker build -f research/docker/Dockerfile.spike \
	  --build-arg BASE=$(IMAGE_DEV_VK) --build-arg SPIKE=$(SPIKE) \
	  -t $(IMAGE_SPIKE)-vk research/
	docker run --rm --device /dev/dri:/dev/dri \
	  --group-add $$(getent group render | cut -d: -f3) \
	  --group-add $$(getent group video  | cut -d: -f3) \
	  -e XDG_RUNTIME_DIR=/tmp \
	  $(IMAGE_SPIKE)-vk /work/build/$(SPIKE)/$(SPIKE)

## Build an analysis tool, e.g. make tool TOOL=expert_trace
tool: dev-vk
	docker build -f research/docker/Dockerfile.llama --build-arg BASE=$(IMAGE_DEV_VK) -t $(IMAGE_LLAMA) research/
	docker build -f research/docker/Dockerfile.tool \
	  --build-arg BASE=$(IMAGE_LLAMA) --build-arg TOOL=$(TOOL) -t $(IMAGE_TOOL) research/

# =============================================================================
# Cleanup
# =============================================================================
## Remove only the verification images this project built; keeps the product image
clean-docker:
	-docker rmi -f $(IMAGE_SPIKE) $(IMAGE_SPIKE)-vk $(IMAGE_LLAMA) $(IMAGE_TOOL) \
	  $(IMAGE_DEV) $(IMAGE_DEV_VK) 2>/dev/null
	docker images --filter "reference=$(IMAGE_PREFIX)*"

images:
	docker images --filter "reference=$(IMAGE_PREFIX)*"
