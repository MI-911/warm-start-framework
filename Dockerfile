FROM python:3.7
# Adding requirements first allows Docker to cache the install of requirements
ADD requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
ADD . /app/
ENTRYPOINT ["python", "entrypoint.py"]
