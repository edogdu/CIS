FROM apache/spark:3.5.1

USER root


RUN mkdir -p /home/spark/.ivy2 && chown -R spark:spark /home/spark/.ivy2

WORKDIR /opt/spark/work-dir

ENV JAVA_TOOL_OPTIONS="-Duser.name=spark -Duser.home=/home/spark"

USER spark
