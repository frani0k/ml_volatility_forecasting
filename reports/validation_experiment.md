# Eksperyment: wpływ liczby obserwacji treningowych na jakość modelu

## 1. Pytanie badawcze

Jak zmienia się jakość prognoz modelu Ridge, gdy zamiast wszystkich dostępnych obserwacji treningowych wykorzystujemy losową próbę obejmującą 80% tych obserwacji, pozostawiając zbiór testowy bez zmian?

## 2. Hipoteza

Losowe pominięcie 20% obserwacji treningowych nie spowoduje dużej zmiany błędu MAE na ustalonym zbiorze testowym.

Hipoteza ma charakter eksploracyjny. Eksperyment nie został zaprojektowany do formalnego wykazania równoważności modeli.

## 3. Metodologia

- **Dane:** historyczne notowania SPY z lat 2015–2025.
- **Zmienna docelowa:** pierwiastek z sumy kwadratów stóp zwrotu z kolejnych pięciu sesji.
- **Model:** `StandardScaler` oraz `Ridge(alpha=1.0)`.
- **Metryka oceny:** MAE, czyli średni błąd bezwzględny.
- **Pełny zbiór treningowy:** 2187 obserwacji.
- **Stały zbiór testowy:** 549 obserwacji, od 17 października 2023 r. do 23 grudnia 2025 r.
- **Liczba powtórzeń:** 30 losowych prób treningowych.
- **Rozmiar każdej próby:** 80% pełnego zbioru treningowego.
- **Ziarna losowości:** liczby całkowite od 0 do 29.

W każdym powtórzeniu zbiór testowy pozostaje identyczny. Losowanie odbywa się wyłącznie w obrębie danych dostępnych przed rozpoczęciem okresu testowego.

Skaler jest dopasowywany osobno do każdej próby treningowej za pomocą potoku `scikit-learn`.

## 4. Wyniki

| Metryka                             |      Wynik |
| ----------------------------------- | ---------: |
| MAE przy pełnym zbiorze treningowym | 0,00630634 |
| Średnie MAE z losowych prób         | 0,00630848 |
| Odchylenie standardowe MAE          | 0,00004281 |
| Średnia względna różnica MAE        |    +0,034% |
| Mediana względnej różnicy MAE       |    -0,004% |
| Próby z niższym MAE niż model pełny |    15 z 30 |

Względną różnicę obliczono według wzoru:

`(MAE próby − MAE pełnego modelu) / MAE pełnego modelu × 100%`

Wartość dodatnia oznacza większy błąd niż w modelu trenowanym na pełnym zbiorze.

## 5. Interpretacja

Średnie MAE uzyskane w 30 losowych próbach było o około 0,034% wyższe niż błąd modelu trenowanego na wszystkich dostępnych obserwacjach. Mediana różnicy była nieznacznie ujemna, a 15 z 30 prób osiągnęło niższe MAE niż model pełny.

Wyniki nie wskazują na wyraźne i konsekwentne pogorszenie jakości prognoz po losowym pominięciu 20% obserwacji treningowych w tym konkretnym eksperymencie.

Pojedyncza próba nie wystarczałaby do sformułowania ogólnego wniosku. Powtórzenie eksperymentu pokazało, że wynik zależy również od wyboru obserwacji treningowych.

## 6. Ograniczenia

1. Wszystkie próby wykorzystują ten sam instrument finansowy i ten sam okres testowy.
2. Trzydzieści powtórzeń oznacza trzydzieści wyborów próby treningowej, a nie trzydzieści niezależnych okresów rynkowych.
3. Eksperyment bada wrażliwość na dobór danych treningowych. Nie rozstrzyga bezpośrednio, czy losowy podział na trening i test jest lepszy od podziału chronologicznego.
4. Wyniki nie dowodzą statystycznej równoważności modeli.
5. Rezultaty mogą być inne dla innych instrumentów, okresów, zestawów cech i konfiguracji modelu.

## 7. Odtwarzalność

Eksperyment można uruchomić poleceniem:

`python src/validation_comparison.py`

Wyniki poszczególnych prób zapisywane są w pliku `reports/validation_subset_runs.csv`, podsumowanie w `reports/validation_comparison.csv`, a histogram w `reports/figures/validation_comparison.png`.

Zastosowano ustalone ziarna losowości oraz stały chronologiczny zbiór testowy.
