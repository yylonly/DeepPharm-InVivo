nohup java -Xms2G -Xmx7G -Dorg.bytedeco.javacpp.maxbytes=8G -Dorg.bytedeco.javacpp.maxphysicalbytes=8G -jar transferlearning-0.0.1-SNAPSHOT-all.jar src/main/resources/train_dataset.csv  src/main/resources/valid_dataset.csv 200 > /root/transferlearning/transferlearningi.out &