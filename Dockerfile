FROM pytorch/pytorch:latest
RUN pip install pandas

COPY script.sh /script.sh
RUN chmod +x /script.sh && /script.sh
