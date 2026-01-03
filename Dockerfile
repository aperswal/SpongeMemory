# Build stage
# Using nightly because hnsw_rs dependencies require Rust edition 2024
FROM rustlang/rust:nightly-alpine AS builder

RUN apk add --no-cache musl-dev pkgconfig openssl-dev openssl-libs-static

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/sponge-core/Cargo.toml ./crates/sponge-core/
COPY crates/sponge-server/Cargo.toml ./crates/sponge-server/

# Create dummy source files for dependency caching
RUN mkdir -p crates/sponge-core/src crates/sponge-server/src && \
    echo "pub fn dummy() {}" > crates/sponge-core/src/lib.rs && \
    echo "fn main() {}" > crates/sponge-server/src/main.rs

# Build dependencies (this layer is cached)
RUN cargo build --release --package sponge-server

# Copy actual source
COPY crates/sponge-core/src ./crates/sponge-core/src
COPY crates/sponge-server/src ./crates/sponge-server/src

# Touch to invalidate cache and rebuild
RUN touch crates/sponge-core/src/lib.rs crates/sponge-server/src/main.rs

# Build release binary
RUN cargo build --release --package sponge-server

# Runtime stage
FROM alpine:3.20

RUN apk add --no-cache ca-certificates

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/sponge /app/sponge

# Create data directory
RUN mkdir -p /data

ENV SPONGE_STORAGE_DATA_PATH=/data

EXPOSE 8080

ENTRYPOINT ["/app/sponge"]
