Subject : With a question and contexts as an input, create an algorithm that will predit which context corresponds the most to the question. 
In order to solve this issue, a simple vectorisation method has been implemented on the question and the contexts. 
Once vectorisation is done a cosine similarity between the question and all the contexts is established. 
The algorithm will output the context that has the highest cosine value with the question vector. 

In order to build the image from the Dockerfile : 

> docker image build . -t squad_predict_image 

In order to run the container and run predict.py file : 

> docker run --name illuin_predict_SQUAD -it -p 80:80 squad_predict_image bash


> python3 predict.py 

The results are printed on terminal and stored under the file name latest_prediction.csv
