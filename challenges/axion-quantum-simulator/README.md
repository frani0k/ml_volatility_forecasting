# Axion Research Challenge 2: symulator bramek kwantowych

## Cel projektu

Projekt zawiera edukacyjny symulator obwodów kwantowych oparty na Pythonie i bibliotece NumPy. Reprezentuje stan rejestru o dowolnej dodatniej liczbie kubitów, na jaką pozwalają dostępna pamięć i zasoby komputera. Umożliwia stosowanie wybranych bramek oraz losowanie wyników pomiaru. Symulacja jest wykonywana klasycznie; program nie komunikuje się z fizycznym komputerem kwantowym.

## Wektor stanu i kolejność kubitów

Stan `n` kubitów to jednowymiarowy wektor NumPy długości `2**n`, zawierający zespolone amplitudy stanów bazowych. Wektor musi mieć długość będącą potęgą dwóch, skończone elementy i normę równą 1. Można utworzyć stan początkowy przez `QuantumState.zero(n)` albo przekazać własny, znormalizowany wektor do `QuantumState`.

Kolejność bitów jest od najbardziej znaczącego do najmniej znaczącego: kubit `q0` odpowiada skrajnemu lewemu bitowi, `q1` następnym bitom, i tak dalej. Indeksy kubitów zaczynają się od zera. Dla dwóch kubitów elementy wektora odpowiadają kolejno stanom `|00>`, `|01>`, `|10>`, `|11>`. Pomiar całego rejestru zwraca bitstring w tej samej kolejności, o długości równej liczbie kubitów.

Prawdopodobieństwo otrzymania danego stanu bazowego wynosi kwadrat modułu jego amplitudy: `|amplituda|**2`. Suma prawdopodobieństw wynosi 1; właściwość `probabilities` zwraca te wartości.

## Obsługiwane bramki

- **X** zamienia `|0>` na `|1>` i `|1>` na `|0>`.
- **Z** pozostawia `|0>` bez zmian, a amplitudzie `|1>` zmienia znak, czyli fazę o `π`.
- **H** tworzy lub łączy równe superpozycje stanów bazowych.
- **Ry(theta)** obraca stan wokół osi Y o kąt `theta` podany w radianach.
- **CNOT** odwraca kubit docelowy, gdy kubit kontrolny ma wartość `1`.

Bramki X, Z, H i Ry można przekazać do `QuantumState.apply(gate, qubit)`, gdzie `qubit` jest dowolnym prawidłowym indeksem w rejestrze. Metoda zwraca nowy stan i stosuje bramkę bezpośrednio do wektora stanu: nie buduje macierzy operatora o rozmiarze `2**n × 2**n`. Bramkę CNOT stosuje `QuantumState.apply_cnot(control, target)`. Oba indeksy mogą wskazywać dowolne różne kubity tego samego rejestru.

Dla CNOT na dwóch kubitach kolejność bazowa to `|00>`, `|01>`, `|10>`, `|11>`; przy kontroli na `q0` i celu na `q1` bramka pozostawia `|00>` i `|01>` bez zmian oraz zamienia `|10>` z `|11>`.

## Pomiar i losowość

`QuantumState.measure(rng=None)` losuje cały rejestr zgodnie z rozkładem `probabilities` i zwraca bitstring o `n` bitach, na przykład `"001"`. Pomiar całego rejestru nie zmienia obiektu stanu. Argument `rng` może być ziarnem typu `int` albo obiektem `numpy.random.Generator`; użycie tego samego ziarna pozwala powtórzyć wynik. Jeśli `rng` pominięto, NumPy tworzy generator korzystający z domyślnego źródła losowości.

`QuantumState.measure_qubit(qubit, rng=None)` mierzy wskazany kubit, zwraca `0` albo `1` i aktualizuje stan, usuwając amplitudy niezgodne z wynikiem oraz normalizując pozostałe.

## Interferencja kwantowa

Zmiana fazy amplitudy nie zmienia prawdopodobieństwa jej bezpośredniego pomiaru: amplitudy `a` i `-a` mają taki sam kwadrat modułu. Faza wpływa jednak na interferencję przy kolejnych bramkach. Dla stanu `|0>` sekwencja H–H daje `H H|0> = |0>`, a sekwencja H–Z–H daje `H Z H|0> = |1>`. Po pierwszej bramce H oba wyniki mają prawdopodobieństwo `1/2`; Z odwraca fazę składnika `|1>`, a końcowa H powoduje, że wkłady amplitud wzmacniają się dla jednego wyniku i znoszą dla drugiego.

Stan Bella powstaje przez H na `q0`, a następnie CNOT z `q0` jako kontrolą i `q1` jako celem:

```text
|00> --H(q0)--> (|00> + |10>)/√2 --CNOT(q0, q1)--> (|00> + |11>)/√2
```

Jego amplitudy to `[1/√2, 0, 0, 1/√2]`, a prawdopodobieństwa wyników to `[1/2, 0, 0, 1/2]`. Pomiar daje więc wyłącznie `"00"` albo `"11"`.

Trójkubitowy stan GHZ można przygotować przez H na `q0` oraz CNOT z `q0` do `q1` i z `q0` do `q2`. Otrzymany stan to `(|000> + |111>)/√2`, więc pomiar całego rejestru zwraca `"000"` albo `"111"`.

## Instalacja i uruchamianie

Wymagane są Python 3 i NumPy. W PowerShell, z katalogu głównego repozytorium, można utworzyć środowisko wirtualne i zainstalować zależność:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy
```

Aby uruchomić przykład:

```powershell
.\.venv\Scripts\python.exe challenges/axion-quantum-simulator/example.py
```

Przykład wypisuje amplitudy i prawdopodobieństwa stanu Bella, pomiar z ziarnem `7` oraz wektor po sekwencji H–Z–H.

Testy symulatora uruchamia się poleceniem:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s challenges/axion-quantum-simulator -p "test_*.py" -v
```

Testy obejmują rejestry od jednego do czterech kubitów, bramki stosowane na różnych pozycjach, pomiar, stan Bella, stan GHZ oraz walidację nieprawidłowych stanów i indeksów.

## Ograniczenia

Symulator przechowuje `2**n` amplitud, dlatego wymagania dotyczące pamięci i czasu rosną wykładniczo wraz z liczbą kubitów. Nie ma sztywnego limitu liczby kubitów w API, ale praktyczny limit wyznaczają zasoby komputera. Program nie modeluje szumu, dekoherencji ani ograniczeń urządzeń. Udostępnia ogólne stosowanie bramek jedno-kubitowych X, Z, H i Ry oraz CNOT z wybranymi indeksami kontroli i celu; nie ma interfejsu do dowolnych bramek wielokubitowych. Jest to klasyczna symulacja matematyczna, a nie wykonywanie obwodu na rzeczywistym sprzęcie kwantowym.
