proj3: main.cpp bitmap.o bayesian.o
	g++ -std=c++11 -g -Wall main.cpp bitmap.o bayesian.o -o proj3

main.o: main.cpp
	g++ -std=c++11 -g -Wall -c main.cpp
bitmap.o: bitmap.cpp
	g++ -std=c++11 -g -Wall -c bitmap.cpp -o bitmap.o
bayesian.o: BayesianHealper.h BayesianHealper.cpp
	g++ -std=c++11 -g -Wall -c BayesianHealper.cpp -o bayesian.o

clean:
	rm proj3 main.o bitmap.o
