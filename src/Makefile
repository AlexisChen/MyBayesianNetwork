proj3: main.o bitmap.o bayesian.o
	g++ -o proj3 main.o bitmap.o bayesian.o
main.o: main.cpp
	g++ -std=c++11 -c main.cpp
bitmap.o: bitmap.cpp
	g++ -std=c++11 -c bitmap.cpp
bayesian.o: BayesianHealper.h BayesianHealper.cpp
	g++ -std=c++11 -g -Wall -c BayesianHealper.cpp -o bayesian.o

clean:
	rm proj3 main.o bitmap.o
