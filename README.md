# SANTA

Hi, this is the code of our paper "SANTA: Separate Strategies for Inaccurate and Incomplete Annotation Noise in Distantly-Supervised Named Entity Recognition" accepted by ACL 2023 Findings. Our paper is available here.

News:

Accepted by ACL 2023 Findings. 2023.05

Code released at Github. 2023.08

Preparation
Download different pretrained LMs into resource/ .
Use environments.yaml to get the right environments.
Reproduce results
For EC:

sh train_EC.sh

For Webpage:

sh train_NEWS.sh

For OntoNotes5.0:

sh train_onto.sh

For BC5CDR:

sh train_bc5cdr.sh

We got our results in single A40.
