import numpy as np
import random

# HELPER ACTIVATION FUNCTIONS
#
# Pomocnicze funkcje aktywacji które są używane poniżej. Tak naprawdę najważniejsze są te, które rysowałem
# na tablicy w trakcie warsztatów czyli: relu, softplus_deriv, sigmoid i sigmoid_deriv.
# Pamiętamy że dla relu nie ma pochodnej w punkcie 0 dlatego przybliża się ją pochodną funkcji softplus i dlatego nie ma
# funkcji relu_deriv tylko mamy softplus_deriv (która sama w sobie jest identyczna jak funkcja sigmoid(x)`
def reLU(x):
    return np.maximum(0, x)


def softplus_deriv(x):
    return sigmoid(x)


def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))
# KONIEC HELPERÓW

# Funkcja pomocnicza które odpowiada za aktualizację wag podczas `backpropagation`. Przyjmuje ona dwa argumenty:
# `weights` - "starą" macierz wag
# `deltas`  - wektor zawierający wyniki pochodnych funkcji aktywacji, które mówią nam jak bardzo należy zmienić wagi.
#             Inaczej mówiąc ten wektor zawiera informację jak bardzo dana waga miała wpływ na błąd sieci i jak mocno
#             daną wagę należy zmienić
def update_weights(weights, deltas):
    # tworzymy sobie nową macierz wag, która początkowo jest kopią starej
    new_weights = np.array(weights)

    # każdą wagę aktualizujemy za pomocą uprzednio wyliczonych "delt" które przechowujemy w wektorze `deltas`
    for x in range(len(weights[0])):
        # dzięki temu super operatorowi możemy zmieniać wartości w całych kolumnach tablicy 2d na raz <mind_blown>
        new_weights[:,x] += learning_rate * deltas[x] * bias

    # zwracamy macierz zaktualizowanych wag
    return new_weights


# pomcnicza funkcja która mapuje nam wartości tekstowe takie jak np "Iris-setosa" na wektory liczbowe np "[0, 1, 0]"
def get_target(text_target):
    if text_target.strip() == 'Iris-virginica':
        return np.array([1, 0, 0])
    elif text_target.strip() == 'Iris-setosa':
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])


# hyperparameters - parametry naszej sieci, odpowiednio: współczynnik uczenia oraz bias (którego de facto nie używamy
# w naszej sieci dlatego jego wartość jest ustawiona na `1`. Gdybyśmy chcieli użyć tego parametru to sieć byłaby
# o wiele bardziej skuteczna ale jednocześnie skomplikowałoby to i tak już nie łatwą sytuację, więc dla prostoty
# przykładu parametr `bias` po prostu pominiemy)
learning_rate = 0.01
bias = 1

# wczytujemy dane do pamięci
data = open('iris.data').readlines()

dataset = []
for line in data:
    linesplit = line.split(',')
    dataset.append(linesplit)

# mieszamy dane losowo, po to aby nasz model nie nauczył się kolejności występowania sampli, które w naszym pliku
# `iris.data` są ładnie posortowane po gatunku kwiatka
random.shuffle(dataset)

# dzielimy zbiór na treningowy i testowy w proporcjach 4:1 - ta proporcja jest wybrana dość arbitralnie, ale zazwyczaj
# staramy się tworzyć zbiory tak, aby treningowy stanowił 80% całości, a pozostałe 20% testowy
trainset = dataset[:120]
testset = dataset[120:]

# kolejne parametry naszej sieci, odpowiednio:
# `n_input`  - ilość neuronów wejściowych. W naszym przypadku jest ich 4, ponieważ każdy kwiatek jest opisany 4 liczbami
# `n_hidden` - ilość neuronów ukrytych. Tą wartością można manipulować aby otrzymać odpowiednio dobre wyniki. W naszym
#              przypadku liczba 4 jest odpowiednia
# `n_output` - ilość neuronów wyjściowych. W naszym przypadku jest ich 3, ponieważ każdy kwiatek chcemy przypisać do
#              jednej z trzech klas
n_input = 4
n_hidden = 4
n_output = 3


# macierze wag, odpowiednio:
# `W1` - macierz wag łącząca ze sobą neurony wejściowe i ukryte. Ma wymiary 4x4
# `W2` - macierz wag łącząca ze sobą neurony ukryte i wyjściowe. Ma wymiary 3x4
#
# Inicjalizujemy macierze małymi liczbami z zakresu <0, 1> z rozkładu Dirichleta. Rozkład ten charakteryzuje się tym, że
# liczby wygenerowane przy jego pomocy będą się sumowały do 1. Czyli jak generujemy sobie 4 liczby to ich suma będzie
# równa 1.
# Równie dobrze można było użyć rozkładu Normalnego (Gaussa) - chodzi o to żeby zainicjalizować wagi małymi, losowymi
# wartościami
W1 = np.array([np.random.dirichlet(np.ones(n_hidden), size=1)[0] for i in range(n_input)])
W2 = np.array([np.random.dirichlet(np.ones(n_output), size=1)[0] for i in range(n_hidden)])

# główna pętla która "biega" przez 500 "epok"
for epoch in range(500):

    print('epoch %d' % epoch)

    # w każdej epoce przelatujemy przez cały zbiór treningowy
    for sample in trainset:

        # tworzymy wektor wejściowy, konwertując wartości zaczytane z pliku na liczby, przy pomocy funkcji `float()`
        INPUT = np.array([float(sample[i]) for i in range(n_input)])
        # tworzymy wektor wyjściowy, którego oczekujemy na wyjściu od sieci. Korzystamy z naszego helpera `get_target()`
        TARGET = get_target(sample[n_input])

        # Forward pass - czyli forward propagation, moment w którym "przepuszczamy" przez sieć naszego sampla od lewej
        # do prawej.
        #
        # `H` - to wektor, w którym trzymamy wartości obliczone w neuronach ukrytych w wyniku pomnożenia wektora wejść
        #       przez macierz wag `W1`. Stosujemy tu funkcję aktywacji `reLU()`, która "po prostu działa dobrze" w
        #       neuronach ukrytych, co zostało dowiedzione empirycznie przez wiele modeli.
        # `O` - to wektor wyjściowy, w którym trzymamy wartości obliczone w neuronach wyjściowych w wyniku pomnożenia
        #       wektora ukrytego przez macierz wag `W2`. Stosujemy tu funkcję aktywacji `sigmoid()`, ponieważ ona nie
        #       zwraca nam wartości dyskretnych (np. 0 lub 1) tylko zwraca wartości ciągłe - prawdopodobieństwa
        #       przynależności danego sampla do konkretnej klasy. Ten wektor to wyjście naszej sieci. Na jego podstawie
        #       możemy policzyć błąd sieci, czyli porównać go z wektorem `TARGET` którego oczekiwaliśmy
        H = np.array([reLU(INPUT.dot(W1[:,x])) for x in range(n_hidden)])
        O = np.array([sigmoid(H.dot(W2[:,x])) for x in range(n_output)])

        # BACKPROPAGATION - czyli moment w trakcie uczenia, gdzie obliczamy błąd naszej sieci i propagujemy go wstecz
        # aktualizując wagi.
        #
        # wektor błędu na wyjściu naszej sieci czyli w naszym przypadku po prostu różnica między oczekiwanym wyjściem, a
        # faktycznym.
        # Np. chcieliśmy, aby sieć w przypadku wejścia `[4.8, 3.4, 1.6, 0.2]` zwróciła nam `[0, 1, 0]`,
        # a tymczasem zwróciła nam `[0.32, 0.5, 0.18]`. Musimy więc policzyć błąd naszej sieci. I tutaj znowu,
        # jest wiele funkcji liczenia błędu sieci, które działają lepiej niż w naszym przykładzie, ale dla prostoty
        # użyjemy zwyczajnego odejmowania czyli wyliczamy sobie różnicę między targetem a outputem.
        O_e = TARGET - O
        # wektor błędu neuronów ukrytych, który po prostu jest wynikiem mnożenia błędu na wyjściu przez macierz wag `W2`
        H_e = O_e.dot(W2.T)

        # obliczamy sobie "delty" czyli wartości, które nam mówią jak bardzo należy zaktualizować poszczególne wagi.
        # Do ich obliczenia używamy pochodnych funkcji aktywacji (mówiłem na warsztatach dlaczego - pochodna nam mówi
        # jak bardzo funkcja pierwotna zmienia się w danym punkcie, a więc w tym przypadku pochodna nam powie jak bardzo
        # należy zmienić wartość wagi w taki sposób żeby wynik funkcji pierwotnej zbliżył się do poprawnego).
        #
        # `O_d` - wektor delt dla wag `W2` (relacja output - hidden). Używamy pochodnej funkcji aktywacji na wyjściu,
        #         czyli `sigmoid()` - `sigmoid_deriv()`
        # `H_d` - wektor delt dla wag `W1` (relacja hidden - input). Używamy pochodnej funkcji aktywacji w warstwie
        #         ukrytej, ale ponieważ nasza warstwa ukryta używa `reLU()` (a wiemy że nie ma ona pochodnej w zerze),
        #         to przybliżamy sobie jej pochodną za pomocą funkcji `softplus_deriv()`
        O_d = np.array([(sigmoid_deriv(H.dot(W2[:,x])) * O_e[x]) for x in range(n_output)])
        H_d = np.array([(softplus_deriv(INPUT.dot(W1[:,x])) * H_e[x]) for x in range(n_hidden)])

        # nadpisujemy stare wagi nowymi, używając wcześniej zdefiniowanego helpera `update_weights()`, odpowiednio:
        # `W2` nadpisujemy używając delt `O_d`
        # `W1` nadpisujemy używając delt `H_d`
        W2 = update_weights(W2, O_d)
        W1 = update_weights(W1, H_d)

        # I tak się kończy jedna iteracja przez sieć gdzie mamy:
        #
        # 1. Forward pass
        # 2. Obliczenie błędu
        # 3. Backpropagation
        # 4. Repeat
        ###


# Funkcja pomocnicza która po prostu sprawdza jak działa nasza wytrenowana sieć. Ponieważ w trakcie treningu używaliśmy
# sporo ułatwień dla prostoty przykładu to nasza sieć będzie miała nie za dobrą skuteczność na poziomie 60-67% -
# prawdopodobnie nauczy się dwóch klas, a trzecią będzie failować. Gdybyśmy zastosowali poprawnie współczynnik `bias`
# oraz lepszą funkcję błędu to wynik sięgnąłby około 98-99% (w pliku `mlp.py` znajdziecie sieć neuronową stworzoną
# z wykorzystaniem TensorFlow, gdzie o wiele łatwiej buduje się takie modele i uwzględnia ona brakujące elementy)
def classify(testset):
    # iterujemy przez cały zbiór testowy
    for sample in testset:
        # Tworzymy sobie wektor wejściowy tak samo jak wyżej
        I = np.array([float(sample[i]) for i in range(n_input)])
        # Tworzymy wektor oczekiwany tak samo jak wyżej
        TARGET = get_target(sample[n_input])

        # Robimy Forward pass tak samo jak wyżej, stosując te same funkcje aktywacji, a więc
        # wektor, w którym trzymamy wartości obliczone w neuronach ukrytych
        H = np.array([reLU(I.dot(W1[:,x])) for x in range(n_hidden)])
        # wektor wyjściowy, który składa się z trzech liczb - prawdopodobieństw przynależności do konkretnej klasy
        O = np.array([sigmoid(H.dot(W2[:,x])) for x in range(n_output)])

        # pomocniczy print line który mówi że dla danego sampla, powinien być taki output i mówi czy dobrze jest czy źle
        print('for this sample', I, 'it should be', TARGET, 'but got', O, 'so', np.argmax(TARGET) == np.argmax(O))


# wywołanie funkcji xD
classify(testset)
