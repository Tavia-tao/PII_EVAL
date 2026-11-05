# 仅提供镜像构建定义；本次开发不会执行构建或启动容器。
FROM python:3.11-slim
WORKDIR /app
ENV DEEPEVAL_TELEMETRY_OPT_OUT=YES \
    DEEPEVAL_UPDATE_WARNING_OPT_OUT=YES \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv==0.12.2
COPY pyproject.toml uv.lock README.md ./
COPY conf ./conf
COPY src ./src
RUN uv sync --frozen --no-dev --python /usr/local/bin/python --no-managed-python
ENTRYPOINT ["pii-eval"]
CMD ["--help"]
