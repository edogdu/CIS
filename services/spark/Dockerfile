FROM bitnami/spark:3.5

USER root
# Create a named user with the same UID Bitnami often uses (1001)
RUN useradd -m -u 1001 -s /bin/bash spark || true \
 && mkdir -p /tmp/.ivy2 \
 && chown -R spark:spark /tmp /tmp/.ivy2 /home/spark

ENV JAVA_TOOL_OPTIONS="-Duser.name=spark -Duser.home=/home/spark"
USER spark
