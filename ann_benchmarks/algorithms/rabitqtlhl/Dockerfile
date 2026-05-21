FROM ann-benchmarks:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir "numpy<2"

COPY run_algorithm.py /tmp/run_algorithm.py
COPY ann_benchmarks /home/app/ann_benchmarks

ENV PYTHONPATH=/home/app
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV RABITQTLHL_LIB=/opt/rabitqtlhl/librabitqtlhl.so

WORKDIR /home/app/ann_benchmarks/algorithms/rabitqtlhl
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j
RUN mkdir -p /opt/rabitqtlhl && \
    cp /home/app/ann_benchmarks/algorithms/rabitqtlhl/librabitqtlhl.so /opt/rabitqtlhl/librabitqtlhl.so

RUN python3 -c "from ann_benchmarks.algorithms.rabitqtlhl import RabitQTLHL; print('rabitqtlhl import ok')"
RUN test -f /tmp/run_algorithm.py

WORKDIR /home/app
ENTRYPOINT ["python3", "-u", "/tmp/run_algorithm.py"]
