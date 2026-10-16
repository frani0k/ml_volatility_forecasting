# Czy dłuższa historia zmienności pomaga w prognozowaniu SPY?

## Streszczenie

W tym projekcie sprawdzam, czy model uczenia maszynowego może prognozować krótkoterminową zmienność ETF-u SPY dokładniej niż prosta metoda bazowa. Następnie badam, czy zmienność z ostatnich 20 sesji wnosi użyteczną informację oraz czy regresja Ridge daje praktyczną przewagę nad zwykłą regresją liniową.

Na głównym zbiorze testowym Ridge osiągnął MAE równy **0,006306**, wobec **0,008005** dla metody bazowej — błąd był niższy o **21,22%**. Usunięcie cechy `Volatility_20d` zwiększyło MAE o **4,28%**, a pogorszenie wystąpiło również we wszystkich czterech dodatkowych oknach chronologicznych. Jednocześnie Ridge i zwykła regresja liniowa osiągnęły niemal identyczne wyniki. Przedziały ufności z blokowego bootstrapu dla różnicy ich błędów obejmowały zero, więc wyniki nie dają przekonujących dowodów na praktyczną przewagę Ridge. W dodatkowym eksperymencie 30 losowych prób obejmujących 80% obserwacji treningowych dało średnie MAE o 0,034% wyższe niż model trenowany na pełnym zbiorze.

To eksperyment prognostyczny na historycznych danych SPY, a nie strategia inwestycyjna ani dowód, że uzyskane wyniki będą powtarzać się na innych aktywach.

## 1. Pytanie badawcze

Czy informacje o niedawnej zmienności rynku pozwalają dokładniej prognozować wielkość ruchów cen SPY w kolejnych pięciu sesjach giełdowych?

Nie próbuję przewidywać, czy cena wzrośnie, czy spadnie. Celem jest oszacowanie skali przyszłych ruchów. Po porównaniu modelu z prostą metodą bazową sprawdzam też dwie bardziej szczegółowe kwestie:

1. Czy cecha `Volatility_20d` wnosi użyteczną informację ponad pozostałe cechy?

2. Czy regularyzacja Ridge poprawia prognozy względem zwykłej regresji liniowej?
3. Jak wrażliwy jest wynik na losowy dobór obserwacji treningowych przy niezmienionym zbiorze testowym?

Hipotezy i dodatkowe testy powstały w toku eksploracyjnego projektu, po wstępnym porównaniu modelu z baseline'em. Nie należy więc traktować ich jako hipotez zarejestrowanych przed obejrzeniem wyników.

## 2. Hipotezy

### H1 — znaczenie zmienności z 20 sesji

**Usunięcie `Volatility_20d` z modelu Ridge pogorszy jakość prognoz mierzoną przez MAE.**

Porównuję pełny model z modelem, w którym usuwam wyłącznie tę cechę. Pozostałe cechy, parametr `alpha`, skalowanie i podział danych pozostają bez zmian. Hipotezę wspiera wzrost błędu na głównym teście oraz powtarzalność tego wyniku w kolejnych oknach chronologicznych.

### H2 — przewaga regularyzacji Ridge

**Regresja Ridge osiągnie niższy MAE niż zwykła regresja liniowa, gdy oba modele używają tych samych cech i tych samych podziałów danych.**

Porównuję oba modele na głównym teście i w czterech oknach chronologicznych. Ponieważ oczekiwana różnica jest niewielka, dodatkowo szacuję niepewność różnicy błędów metodą blokowego bootstrapu.

### H3 — wrażliwość na dobór próby treningowej

**Losowe pominięcie 20% obserwacji treningowych nie spowoduje dużej zmiany MAE na ustalonym zbiorze testowym.**

W eksperymencie zbiór testowy pozostaje stały, a losowanie dotyczy wyłącznie obserwacji z pełnego zbioru treningowego. Hipoteza jest eksploracyjna; procedura nie jest testem statystycznej równoważności.

## 3. Dane

Dane historyczne ETF-u SPY pobrałem z Yahoo Finance za pomocą biblioteki `yfinance`. Zakres obejmuje lata 2015–2025.

- Liczba obserwacji w surowym zbiorze: **2766**.

- Liczba obserwacji po przygotowaniu cech i celu: **2741**.

- Daty w przygotowanym zbiorze są uporządkowane.

- Nie znaleziono zduplikowanych dat ani brakujących wartości w przygotowanym zbiorze.

Wyniki dotyczą jednego instrumentu i jednego zakresu historycznego. Nie zakładam, że będą automatycznie uogólniać się na inne ETF-y, akcje lub okresy rynkowe.

## 4. Definicja celu

Na podstawie cen obliczam dzienne stopy zwrotu:

$$
r_t = \frac{P_t}{P_{t-1}} - 1
$$

Celem prognozy jest pierwiastek z sumy kwadratów stóp zwrotu z kolejnych pięciu sesji:

$$
Y_t = \sqrt{\sum_{i=1}^{5} r_{t+i}^{2}}
$$

Miara ta opisuje łączną wielkość ruchów cen w przyszłym pięciosesyjnym okresie. Jest to **pięciosesyjna miara RSS**, a nie klasyczne odchylenie standardowe ani prognoza kierunku ceny.

Cechy wejściowe korzystają z danych dostępnych najpóźniej w dniu prognozy. Cel wykorzystuje stopy zwrotu z kolejnych pięciu sesji. W głównym podziale pozostawiam pięć obserwacji odstępu między końcem treningu a początkiem testu, aby ograniczyć nakładanie się okresów celu po obu stronach podziału.

## 5. Cechy i modele

Pełny model wykorzystuje pięć cech:

- `Volatility_5d` — zmienność stóp zwrotu z ostatnich 5 sesji,

- `Volatility_20d` — zmienność stóp zwrotu z ostatnich 20 sesji,

- `Mean_Return_5d` — średnia stopa zwrotu z ostatnich 5 sesji,

- `Mean_Return_20d` — średnia stopa zwrotu z ostatnich 20 sesji,

- `Mean_Absolute_Return_5d` — średnia wartość bezwzględnej stopy zwrotu z ostatnich 5 sesji.

Model Ridge korzysta z `alpha = 1.0` i `StandardScaler`. Skalowanie jest dopasowywane wyłącznie na danych treningowych. Ujemne prognozy są ograniczane do zera, ponieważ cel nie może być ujemny.

Jako metodę bazową wykorzystuję pierwiastek z sumy kwadratów stóp zwrotu z poprzednich pięciu sesji. Baseline pozwala ocenić, czy model wnosi coś ponad prostą prognozę opartą na niedawnej historii.

## 6. Protokół ewaluacji

Dane dzielę chronologicznie, bez losowego mieszania obserwacji. Główny eksperyment wykorzystuje:

- **2187** obserwacji treningowych,

- **549** obserwacji testowych,

- pięć obserwacji odstępu między treningiem a testem.

Trening kończy się 2023-10-09, a główny test rozpoczyna się 2023-10-17. Metryką główną jest **MAE (Mean Absolute Error)** — średnia wartość bezwzględnego błędu prognozy. Niższy MAE oznacza dokładniejsze prognozy w tej skali celu.

Dodatkowo przeprowadzam ewaluację w czterech kolejnych oknach chronologicznych. Pierwszy zbiór treningowy stanowi 60% danych, a każde okno testowe około 10%; zbiór treningowy rozszerza się po każdym oknie. Każde okno testowe zawiera 274 obserwacje, a między treningiem i testem pozostaje pięć wierszy odstępu. Te okna nie są niezależnymi próbami: pochodzą z tej samej serii czasowej, a cele dla sąsiednich dat częściowo się pokrywają. Nie traktuję ich jako czterech niezależnych potwierdzeń statystycznych.

## 7. Wynik główny: Ridge kontra metoda bazowa

| Metoda            |      MAE |
| ----------------- | -------: |
| Metoda bazowa     | 0,008005 |
| Ridge — pięć cech | 0,006306 |

Ridge uzyskał na głównym teście MAE niższy o **21,22%** względem baseline'u. To zachęcający wynik, ale sam w sobie nie wyjaśnia, które cechy są przydatne ani czy różnica będzie utrzymywać się w przyszłości.

## 8. Eksperyment H1: usunięcie `Volatility_20d`

Porównałem pełny model Ridge z wariantem pozbawionym wyłącznie cechy `Volatility_20d`.

### Główny test

| Wariant                    |      MAE |
| -------------------------- | -------: |
| Ridge — wszystkie cechy    | 0,006306 |
| Ridge bez `Volatility_20d` | 0,006576 |

Po usunięciu cechy MAE wzrósł o **4,28%** względem pełnego modelu.

### Kolejne okna chronologiczne

| Okres testowy | MAE pełnego Ridge | MAE bez `Volatility_20d` | Zmiana MAE po usunięciu |
| ------------- | ----------------: | -----------------------: | ----------------------: |
| 2021–2022     |          0,007718 |                 0,008008 |                  +3,75% |
| 2022–2023     |          0,005534 |                 0,006091 |                 +10,07% |
| 2023–2024     |          0,004781 |                 0,005009 |                  +4,77% |
| 2024–2025     |          0,007833 |                 0,008108 |                  +3,52% |

We wszystkich czterech oknach usunięcie cechy pogorszyło wynik. Średni wzrost MAE w tych oknach wyniósł **5,53%**.

### Wniosek H1

Wyniki **wspierają hipotezę**, że `Volatility_20d` wnosi użyteczną informację dla tego modelu, zestawu pozostałych cech i badanego okresu. Nie dowodzą jednak związku przyczynowego ani statystycznej istotności. Cechy opisujące zmienność mogą być ze sobą skorelowane, a wyniki pochodzą z jednej serii SPY.

## 9. Dodatkowa ewaluacja Ridge w czasie

| Okres testowy | MAE baseline'u | MAE Ridge | Poprawa względem baseline'u |
| ------------- | -------------: | --------: | --------------------------: |
| 2021–2022     |       0,008768 |  0,007718 |                      11,97% |
| 2022–2023     |       0,007640 |  0,005534 |                      27,57% |
| 2023–2024     |       0,006215 |  0,004781 |                      23,06% |
| 2024–2025     |       0,009792 |  0,007833 |                      20,00% |

Ridge uzyskał niższy MAE niż baseline we wszystkich czterech oknach. Średnia poprawa wyniosła **20,65%**, a mediana **21,53%**. To dodatkowy argument, że model jest wart dalszego badania, ale nie jest to niezależne potwierdzenie H1 — okna pochodzą z tej samej serii i częściowo pokrywają się z głównym okresem testowym.

## 10. Dodatkowa analiza poziomu przyszłej zmienności

Podzieliłem obserwacje głównego testu na 20% przypadków o najwyższej zrealizowanej przyszłej zmienności oraz pozostałe 80%.

| Grupa                    | Liczba obserwacji | MAE baseline'u | MAE Ridge |
| ------------------------ | ----------------: | -------------: | --------: |
| Najwyższe 20% zmienności |               110 |       0,017564 |  0,013916 |
| Pozostałe obserwacje     |               439 |       0,005610 |  0,004400 |

Ridge miał niższy błąd w obu grupach, ale błędy były znacznie większe w okresach o najwyższej przyszłej zmienności. Jest to analiza diagnostyczna po fakcie: grupy definiuje rzeczywisty przyszły cel, którego nie znamy w momencie tworzenia prognozy. Nie jest to więc reguła możliwa do bezpośredniego zastosowania w czasie rzeczywistym.

## 11. Eksperyment H2: Ridge kontra zwykła regresja liniowa

Aby sprawdzić, czy regularyzacja ma znaczenie, porównałem Ridge i zwykłą regresję liniową na tych samych pięciu cechach, z identycznym podziałem danych, skalowaniem i metryką.

### Główny test

| Model            |         MAE |
| ---------------- | ----------: |
| Regresja liniowa | 0,006306482 |
| Ridge            | 0,006306344 |

Poprawa Ridge wyniosła około **0,00218%** — w praktyce wyniki były niemal identyczne.

### Kolejne okna chronologiczne

| Okres testowy | MAE regresji liniowej | MAE Ridge | Poprawa Ridge |
| ------------- | --------------------: | --------: | ------------: |
| 2021–2022     |              0,007730 |  0,007718 |       +0,149% |
| 2022–2023     |              0,005536 |  0,005534 |       +0,040% |
| 2023–2024     |              0,004783 |  0,004781 |       +0,042% |
| 2024–2025     |              0,007832 |  0,007833 |       −0,019% |

Ridge miał niższy MAE w trzech z czterech okien, ale różnice były bardzo małe. W ostatnim oknie minimalnie lepsza była regresja liniowa. Same wyniki punktowe nie dają przekonujących dowodów na praktyczną przewagę Ridge.

## 12. Niepewność różnicy między Ridge a regresją liniową

Ponieważ błędy dla kolejnych dni mogą być zależne, do oszacowania niepewności użyłem **blokowego bootstrapu** na głównym zbiorze testowym. Losowałem bloki kolejnych obserwacji z błędami obu modeli, zachowując parowanie błędów z tych samych dat. Wykonałem 5000 replikacji dla długości bloku 5, 10 i 20 obserwacji, z ustalonym ziarnem generatora `42`.

Różnicę zdefiniowałem jako MAE regresji liniowej minus MAE Ridge. Wartość dodatnia oznacza przewagę Ridge.

| Długość bloku | 95% przedział dla różnicy MAE | 95% przedział względnej poprawy Ridge |
| ------------: | ----------------------------: | ------------------------------------: |
|             5 |     [−0,00000321; 0,00000291] |                    [−0,052%; +0,047%] |
|            10 |     [−0,00000342; 0,00000300] |                    [−0,052%; +0,050%] |
|            20 |     [−0,00000339; 0,00000265] |                    [−0,047%; +0,047%] |

Wszystkie przedziały obejmują zero. W tej analizie nie ma więc przekonujących dowodów, że Ridge przewyższa zwykłą regresję liniową. Nie jest to dowód, że modele są równoważne; oznacza jedynie, że obserwowana różnica jest bardzo mała względem niepewności oszacowanej tą metodą.

Wynik zależy od przyjętej metody bootstrapu i długości bloków. Analiza dotyczy jednego historycznego zbioru testowego i nie gwarantuje zachowania podobnych różnic w przyszłości.

## 13. Eksperyment H3: wrażliwość na dobór próby treningowej

### Pytanie badawcze

Jak zmienia się jakość prognoz Ridge, gdy zamiast wszystkich dostępnych obserwacji treningowych wykorzystujemy losową próbę obejmującą 80% danych, pozostawiając zbiór testowy bez zmian?

### Metoda

Pełny zbiór treningowy liczył 2187 obserwacji, a stały chronologiczny zbiór testowy — 549 obserwacji, od 17 października 2023 r. do 23 grudnia 2025 r.

Wykonano 30 powtórzeń. W każdym losowano 80% obserwacji z pełnego zbioru treningowego, używając ziaren od 0 do 29. Każdy model korzystał z `StandardScaler` oraz `Ridge(alpha=1.0)`. Skaler dopasowywano osobno do każdej próby treningowej. Wszystkie modele oceniano na tych samych obserwacjach testowych.

### Wyniki

| Metryka                             |      Wynik |
| ----------------------------------- | ---------: |
| MAE pełnego zbioru treningowego     | 0,00630634 |
| Średnie MAE losowych prób           | 0,00630848 |
| Odchylenie standardowe MAE          | 0,00004281 |
| Średnia względna różnica MAE        |    +0,034% |
| Mediana względnej różnicy MAE       |    −0,004% |
| Próby z niższym MAE niż model pełny |      15/30 |

Względną różnicę obliczono według wzoru:

`(MAE próby − MAE pełnego modelu) / MAE pełnego modelu × 100%`

Wartość dodatnia oznacza większy błąd niż w modelu trenowanym na pełnym zbiorze.

### Interpretacja

Średnie MAE z 30 losowych prób było o około **0,034% wyższe** niż MAE modelu trenowanego na wszystkich dostępnych obserwacjach. Mediana różnicy była nieznacznie ujemna, a 15 z 30 prób osiągnęło niższy błąd niż model pełny.

Wyniki nie wskazują na wyraźne i konsekwentne pogorszenie jakości prognoz po losowym pominięciu 20% obserwacji treningowych w tym konkretnym eksperymencie.

### Ograniczenia

1. Wszystkie powtórzenia wykorzystują ten sam instrument i ten sam okres testowy.
2. Trzydzieści losowań to trzydzieści wyborów próby treningowej, a nie trzydzieści niezależnych okresów rynkowych.
3. Eksperyment bada wrażliwość na dobór obserwacji treningowych, a nie przewagę losowego podziału nad podziałem chronologicznym.
4. Wyniki nie dowodzą statystycznej równoważności modeli.
5. Rezultaty mogą być inne dla innych instrumentów, okresów, cech i konfiguracji modelu.

Szczegółowy opis eksperymentu znajduje się w `reports/validation_experiment.md`. Poszczególne powtórzenia zapisano w `reports/validation_subset_runs.csv`, podsumowanie w `reports/validation_comparison.csv`, a wykres w `reports/figures/validation_comparison.png`.

## 14. Audyt obliczeń

Skrypt `src/audit_experiment.py` niezależnie odtwarza wszystkie cechy, cel i metodę bazową na podstawie surowych cen. Kończy się błędem, jeśli daty, kompletność danych lub odtworzone wartości nie przejdą kontroli. Audyt wykazał:

- 2741 obserwacji w przygotowanym zbiorze,

- brak zduplikowanych dat,

- prawidłową kolejność dat,

- brak brakujących wartości,

- zerowy błąd przy niezależnym odtworzeniu cech, celu i baseline'u.

Automatyczne testy jednostkowe sprawdzają definicję zwrotu, wyrównanie celu, brak przyszłych danych w cechach, rekonstrukcję baseline'u, podziały chronologiczne, obcinanie ujemnych prognoz i deterministyczność bootstrapu. Audyt i testy sprawdzają spójność implementacji, ale nie są formalnym dowodem braku każdego możliwego rodzaju data leakage. Nadal kluczowe są chronologiczny podział danych i upewnienie się, że cechy są dostępne w momencie prognozy.

## 15. Ograniczenia

1. **Jeden instrument:** badam wyłącznie SPY, więc wynik może nie przenosić się na inne aktywa.

2. **Jedna historia:** dane obejmują lata 2015–2025; zmiany reżimów rynkowych mogą wpływać na wyniki.

3. **Nakładające się cele:** cele dla sąsiednich dat obejmują częściowo te same przyszłe sesje. To ogranicza niezależność obserwacji i ma znaczenie dla niepewności.

4. **Okna nie są niezależne:** dodatkowe okna chronologiczne pochodzą z tej samej serii i częściowo nakładają się na główny okres testowy.

5. **Eksploracyjny charakter:** hipotezy nie zostały zarejestrowane przed analizą wyników; dalsze testy na nowych danych pomogłyby ocenić ich trwałość.

6. **Ograniczony zestaw modeli:** porównuję liniowe modele i nie przeprowadzam szerokiego strojenia hiperparametrów.

7. **Brak testu strategii inwestycyjnej:** niższy błąd prognozy zmienności nie oznacza automatycznie zyskownej strategii po uwzględnieniu kosztów transakcyjnych i ryzyka.

8. **Niepewność bootstrapu:** przedziały zależą od schematu losowania bloków i odnoszą się do jednego historycznego testu.

## 16. Wnioski

Najważniejsze wyniki są następujące:

- Ridge uzyskał MAE niższy o **21,22%** od prostej metody bazowej na głównym teście.

- Usunięcie `Volatility_20d` zwiększyło MAE o **4,28%** na głównym teście i pogorszyło wynik we wszystkich czterech dodatkowych oknach. To wspiera H1 w ramach badanego eksperymentu.

- Ridge i zwykła regresja liniowa uzyskały niemal identyczny wynik. Blokowy bootstrap dał przedziały obejmujące zero, więc nie znalazłem przekonujących dowodów na praktyczną przewagę Ridge.

- Błędy były znacznie większe w obserwacjach o najwyższej zrealizowanej przyszłej zmienności.

Najważniejszą lekcją z projektu jest to, że dobry wynik modelu nie wyjaśnia jeszcze, dlaczego model działa. Ablacja cechy i porównanie modeli pozwoliły sprawdzić konkretne decyzje metodologiczne, a analiza niepewności pomogła uniknąć nadinterpretacji bardzo małej różnicy wyników.

## 17. Dalsze kroki

Najbardziej wartościowe kolejne eksperymenty to:

1. Ocena modelu na innym instrumencie lub na późniejszym okresie, który nie był wykorzystywany przy wyborze modelu.

2. Sprawdzenie stabilności wyników przy innych rozsądnych definicjach celu zmienności.

3. Porównanie z dodatkowymi prostymi baseline'ami i modelami, przy zachowaniu tego samego chronologicznego protokołu.

4. Jeśli projekt miałby służyć do podejmowania decyzji inwestycyjnych, osobna ocena użyteczności prognoz i kosztów wdrożenia — bez zakładania, że sam MAE świadczy o rentowności.

## 18. Pliki projektu

- `src/download_data.py` — pobieranie danych.

- `src/explore_data.py` — wstępna analiza zbioru.

- `src/analyze_returns.py` — analiza stóp zwrotu.

- `src/prepare_features.py` — przygotowanie cech i celu.

- `src/train_model.py` — trening modelu i główna ewaluacja.

- `src/evaluate_results.py` — analiza wyników i wykres prognoz.

- `src/rolling_evaluation.py` — ewaluacja pełnego modelu w kolejnych oknach.

- `src/feature_ablation.py` — eksperyment z usunięciem `Volatility_20d`.

- `src/model_comparison.py` — porównanie Ridge ze zwykłą regresją liniową.

- `src/bootstrap_uncertainty.py` — blokowy bootstrap różnicy błędów.
- `src/validation_comparison.py` — eksperyment wrażliwości na dobór próby treningowej.

- `src/audit_experiment.py` — niezależny audyt obliczeń celu i baseline'u.

- `src/research_utils.py` — wspólne obliczenia cech, celu i podziałów chronologicznych.

- `src/project_paths.py` — ścieżki wyznaczane względem repozytorium.

- `tests/test_research_utils.py` — testy niezmienników metodologicznych w `unittest`.

Najważniejsze zapisane wyniki:

- `reports/test_predictions.csv` — prognozy na głównym zbiorze testowym.

- `reports/rolling_evaluation.csv` — wyniki ewaluacji chronologicznej.

- `reports/feature_ablation_main.csv` i `reports/feature_ablation_rolling.csv` — wyniki ablacji cechy.

- `reports/model_comparison_main.csv` i `reports/model_comparison_rolling.csv` — porównanie regresji liniowej i Ridge.

- `reports/model_comparison_test_predictions.csv` — sparowane predykcje i błędy regresji liniowej oraz Ridge; zapisuje je `src/bootstrap_uncertainty.py`.

- `reports/bootstrap_uncertainty.csv` — wyniki analizy niepewności bootstrapem.
- `reports/validation_comparison.csv` — podsumowanie eksperymentu doboru próby.
- `reports/validation_subset_runs.csv` — wyniki poszczególnych powtórzeń.
- `reports/figures/validation_comparison.png` — histogram wyników eksperymentu.
- `reports/validation_experiment.md` — szczegółowy opis dodatkowego eksperymentu.

- `reports/figures/price_and_volatility.png` i `reports/figures/test_predictions.png` — wykresy diagnostyczne.

## 19. Powtarzalność

Projekt sprawdzono na **CPython 3.14.0**; bezpośrednie zależności i ich wersje znajdują się w `requirements.txt`. Polecenia uruchomieniowe dla Windows PowerShell, w tym testy i audyt, są podane w README. Surowe i przetworzone dane pozostają lokalne i są ignorowane przez Git; pobranie z Yahoo Finance może w przyszłości zwrócić skorygowaną historię.
