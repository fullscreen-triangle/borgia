# Borgia Makefile - Development and Build Automation

.PHONY: help install build test clean run docker format lint check dev docs benchmark

# Default target
help: ## Show this help message
	@echo "Borgia - Revolutionary Probabilistic Cheminformatics Engine"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Installation and setup
install: install-rust install-python ## Install all dependencies

install-rust: ## Install Rust dependencies
	@echo "Installing Rust dependencies..."
	cargo fetch
	cargo build

install-python: ## Install Python dependencies
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -e .

# Development
dev: ## Start development environment
	@echo "Starting development environment..."
	docker-compose up -d postgres redis
	@echo "Waiting for services to be ready..."
	sleep 10
	cargo run

dev-full: ## Start full development environment with all services
	@echo "Starting full development environment..."
	docker-compose up -d

# Building
build: build-rust build-python ## Build all components

build-rust: ## Build Rust components
	@echo "Building Rust components..."
	cargo build --release

build-python: ## Build Python components
	@echo "Building Python components..."
	maturin build --release

# Testing
test: test-rust test-python ## Run all tests

test-rust: ## Run Rust tests
	@echo "Running Rust tests..."
	cargo test --all-features

test-python: ## Run Python tests
	@echo "Running Python tests..."
	pytest python/tests/ -v

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/integration/ -v

test-benchmark: ## Run benchmark tests
	@echo "Running benchmark tests..."
	cargo bench
	pytest tests/benchmarks/ -v --benchmark-only

# Code Quality
format: format-rust format-python ## Format all code

format-rust: ## Format Rust code
	@echo "Formatting Rust code..."
	cargo fmt

format-python: ## Format Python code
	@echo "Formatting Python code..."
	black python/
	isort python/

lint: lint-rust lint-python ## Lint all code

lint-rust: ## Lint Rust code
	@echo "Linting Rust code..."
	cargo clippy --all-features -- -D warnings

lint-python: ## Lint Python code
	@echo "Linting Python code..."
	flake8 python/
	mypy python/

check: ## Run all checks (format, lint, test)
	@echo "Running all checks..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

# Docker
docker-build: ## Build Docker image
	@echo "Building Docker image..."
	docker build -t borgia:latest .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8000:8000 -p 9090:9090 borgia:latest

docker-compose-up: ## Start all services with docker-compose
	@echo "Starting all services..."
	docker-compose up -d

docker-compose-down: ## Stop all services
	@echo "Stopping all services..."
	docker-compose down

docker-compose-logs: ## Show logs from all services
	docker-compose logs -f

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	cargo doc --no-deps --open
	cd python && sphinx-build -b html docs docs/_build/html

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	cd python/docs/_build/html && python -m http.server 8080

# Benchmarking
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	cargo bench --bench molecular_similarity
	cargo bench --bench probabilistic_algorithms
	cargo bench --bench fuzzy_logic_performance

# Profiling
profile: ## Profile the application
	@echo "Profiling application..."
	cargo build --release --features profiling
	perf record -g ./target/release/borgia
	perf report

# Database
db-setup: ## Setup database
	@echo "Setting up database..."
	docker-compose up -d postgres
	sleep 5
	psql -h localhost -U borgia -d borgia -f scripts/init_db.sql

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	sqlx migrate run

db-reset: ## Reset database
	@echo "Resetting database..."
	docker-compose down postgres
	docker-compose up -d postgres
	sleep 5
	$(MAKE) db-setup

# Utilities
run: ## Run the application
	@echo "Running Borgia..."
	cargo run --release

run-api: ## Run the API server
	@echo "Running API server..."
	python -m borgia.api

run-cli: ## Run the CLI
	@echo "Running CLI..."
	cargo run --bin borgia

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -rf dist/
	rm -rf build/
	rm -rf python/borgia.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Monitoring
monitor: ## Start monitoring services
	@echo "Starting monitoring services..."
	docker-compose up -d prometheus grafana

logs: ## Show application logs
	@echo "Showing application logs..."
	tail -f logs/borgia.log

# Data and Models
download-data: ## Download test data
	@echo "Downloading test data..."
	mkdir -p data/molecules
	curl -o data/molecules/test_molecules.sdf "https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/data/cdk2.sdf"

prepare-data: ## Prepare data for processing
	@echo "Preparing data..."
	python scripts/prepare_data.py

# Security
security-audit: ## Run security audit
	@echo "Running security audit..."
	cargo audit
	pip-audit

# Release
release-dry-run: ## Dry run release
	@echo "Dry run release..."
	cargo publish --dry-run

release: ## Release new version
	@echo "Releasing new version..."
	cargo publish

# Environment setup
setup-env: ## Setup environment variables
	@echo "Setting up environment..."
	cp env.example .env
	@echo "Please edit .env file with your configuration"

# Full development setup
setup: install setup-env db-setup download-data ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Run 'make dev' to start the development server"

# CI/CD
ci: ## Run CI pipeline locally
	@echo "Running CI pipeline..."
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) build
	$(MAKE) security-audit 