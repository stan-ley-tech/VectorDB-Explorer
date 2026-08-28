# syntax=docker/dockerfile:1

# Stage 1: compile the C++ engine and its Python bindings. Kept separate
# from the runtime stage so build-essential/cmake never end up in the final
# image - only the compiled vdb_native*.so is copied over.
FROM python:3.12-slim AS engine-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY engine ./engine
RUN cmake -S engine -B engine/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DVDB_BUILD_TESTS=OFF \
        -DVDB_BUILD_PYTHON_BINDINGS=ON \
    && cmake --build engine/build --config Release -j

# Stage 2: the API runtime. Only needs Python and the wheels below - no
# compiler toolchain.
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini

# Optional at runtime: app.engine_bridge falls back to a numpy brute-force
# implementation if this import fails, but copying it in means approximate
# (IVF) search and the indexed side of benchmarks work out of the box.
COPY --from=engine-builder /src/engine/build/bindings/vdb_native*.so ./

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
