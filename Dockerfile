FROM subtle/subtle_synth_tf2_6:2022-08-01

RUN pip3 install --upgrade pip

COPY ./train/requirements_inference.txt /tmp/requirements_inference.txt
RUN pip3 install -r /tmp/requirements_inference.txt

RUN pip3 uninstall -y SimpleITK > /dev/null
RUN wget -N https://com-subtlemedical-dev-public.s3.amazonaws.com/elastix/elastix-centos.tar.gz
RUN tar -zxvf elastix-centos.tar.gz
WORKDIR /opt/elastix-centos
RUN sed -i "s@\/opt\/build@$(pwd)@g" SimpleITK-build/Wrapping/Python/Packaging/setup.py
WORKDIR /opt/elastix-centos/SimpleITK-build/Wrapping/Python
RUN python3 Packaging/setup.py install
WORKDIR /opt

RUN rm -rf elastix*
RUN python3.7 -c "import SimpleITK as sitk; sitk.ElastixImageFilter(); print('SimpleElastix successfully installed');"

ENTRYPOINT ["python3.7", "train/inference.py"]
