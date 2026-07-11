# =============================================================================
# Makefile — thesis-sst
# Shortcut perintah Docker untuk workflow sehari-hari
#
# Cara pakai:
#   make help         — tampilkan semua perintah
#   make build        — build image CPU
#   make jupyter      — buka JupyterLab
#   make preprocess   — jalankan preprocessing
#   make shell        — masuk ke shell container
# =============================================================================

.PHONY: help build build-cuda jupyter preprocess train-test evaluate \
        shell logs clean prune hpc-build

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  thesis-sst — Perintah Docker"
	@echo "  ════════════════════════════════════════"
	@echo ""
	@echo "  SETUP"
	@echo "    make setup          Salin .env.example ke .env (pertama kali)"
	@echo "    make build          Build image Docker CPU (Mac M2 Pro)"
	@echo "    make build-cuda     Build image Docker CUDA (HPC BMKG)"
	@echo ""
	@echo "  WORKFLOW UTAMA"
	@echo "    make preprocess     Jalankan pipeline preprocessing CMEMS + Argo"
	@echo "    make jupyter        Buka JupyterLab di localhost:8888"
	@echo "    make train-test     Uji logika training (3 epoch, CPU)"
	@echo "    make evaluate       Hitung metrik evaluasi semua model"
	@echo ""
	@echo "  DEVELOPMENT"
	@echo "    make shell          Shell interaktif di dalam container"
	@echo "    make logs           Tampilkan log container aktif"
	@echo "    make clean          Hapus container yang berhenti"
	@echo "    make prune          Bersihkan semua image & volume tidak terpakai"
	@echo ""
	@echo "  HPC BMKG"
	@echo "    make hpc-build      Build CUDA image & export ke .tar.gz untuk Singularity"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ .env dibuat dari .env.example — isi dengan kredensial CMEMS Anda"; \
	else \
		echo "ℹ  .env sudah ada, tidak ditimpa"; \
	fi

# ── Build ─────────────────────────────────────────────────────────────────────
build:
	@echo "Building thesis-sst:cpu (Mac M2 Pro mode)..."
	docker build \
		--build-arg TARGET=cpu \
		--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
		-t thesis-sst:cpu \
		.
	@echo "✓ Image thesis-sst:cpu siap"

build-cuda:
	@echo "Building thesis-sst:cuda (HPC BMKG mode, linux/amd64)..."
	docker build \
		--build-arg TARGET=cuda \
		--build-arg BUILD_DATE=$$(date -u +"%Y-%m-%dT%H:%M:%SZ") \
		--platform linux/amd64 \
		-t thesis-sst:cuda \
		.
	@echo "✓ Image thesis-sst:cuda siap"

# ── Workflow utama ────────────────────────────────────────────────────────────
preprocess:
	docker compose up preprocess

jupyter:
	@echo "Membuka JupyterLab di http://localhost:8888 ..."
	docker compose up jupyter

train-test:
	docker compose up train-test

evaluate:
	docker compose up evaluate

# ── Development ───────────────────────────────────────────────────────────────
shell:
	docker compose run --rm shell

logs:
	docker compose logs -f

clean:
	docker compose down --remove-orphans
	docker container prune -f

prune:
	@echo "⚠️  Ini akan menghapus semua container, image, dan volume tidak terpakai."
	@read -p "Lanjutkan? [y/N] " ans; \
	if [ "$$ans" = "y" ] || [ "$$ans" = "Y" ]; then \
		docker system prune -af --volumes; \
	fi

# ── HPC BMKG ──────────────────────────────────────────────────────────────────
hpc-build:
	chmod +x docker/build_for_hpc.sh
	./docker/build_for_hpc.sh
