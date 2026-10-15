# Prognozowanie zmienności SPY

**Pytanie badawcze:** Czy uczenie maszynowe rzeczywiście poprawia prognozowanie zmienności?

Projekt porównuje prostą metodę historycznej zmienności z regresją Ridge na dziennych danych ETF-u SPY. Oprócz wyniku głównego sprawdza znaczenie wybranej cechy, różnice między modelami oraz wrażliwość wyniku na dobór obserwacji treningowych. To eksperyment badawczy, nie porada inwestycyjna ani test strategii.

## Wyniki

Na głównym, chronologicznym zbiorze testowym Ridge osiągnął MAE **0,006306**, a baseline **0,008005** — różnica wyniosła **21,22%** na korzyść Ridge. Usunięcie cechy `Volatility_20d` zwiększyło MAE o **4,28%**; wzrost wystąpił też we wszystkich czterech dodatkowych oknach.

Porównanie Ridge ze zwykłą regresją liniową dało niemal identyczny wynik: MAE wyniosły odpowiednio **0,006306344** i **0,006306482**. Blokowy bootstrap (5000 replikacji, bloki długości 5, 10 i 20) dał przedziały obejmujące zero. Wyniki nie stanowią przekonującego dowodu na praktyczną przewagę Ridge nad regresją liniową ani na równoważność tych modeli.

### Wrażliwość na dobór próby treningowej

Wykonano 30 powtórzeń, w których losowo wybierano 80% z 2187 obserwacji treningowych. Każdy model oceniano na tym samym zbiorze testowym liczącym 549 obserwacji.

| Metryka                             |      Wynik |
| ----------------------------------- | ---------: |
| MAE pełnego zbioru treningowego     | 0,00630634 |
| Średnie MAE losowych prób           | 0,00630848 |
| Odchylenie standardowe MAE          | 0,00004281 |
| Średnia względna różnica MAE        |    +0,034% |
| Mediana względnej różnicy MAE       |    −0,004% |
| Próby z niższym MAE niż model pełny |      15/30 |

W tym eksperymencie losowe pominięcie 20% obserwacji treningowych nie wiązało się z wyraźną zmianą średniego błędu. Wszystkie powtórzenia wykorzystują jednak ten sam okres testowy; nie są to 30 niezależnych testów rynkowych ani dowód statystycznej równoważności.

## Dane i definicja celu

Skrypt pobiera dzienne ceny SPY z Yahoo Finance (`yfinance`) dla okresu 2015-01-01–2025-12-31 z `auto_adjust=True`. Parametr `end` w yfinance jest wyłączny, dlatego skrypt przekazuje `2026-01-01`. Surowe dane zapisuje lokalnie w `data/raw/spy_prices.csv`; pliki danych nie są śledzone przez Git. Yahoo Finance może później korygować historię, więc ponowne pobranie nie musi odtworzyć identycznego pliku.

Dzienna prosta stopa zwrotu:

\[
r*t = \frac{P_t}{P*{t-1}} - 1
\]

Prognozowany cel to pierwiastek z sumy kwadratów zwrotów z kolejnych pięciu sesji:

\[
Y*t = \sqrt{\sum*{i=1}^{5} r\_{t+i}^{2}}
\]

To pięciosesyjna miara RSS, a nie klasyczne odchylenie standardowe. Model wykorzystuje pięć cech obliczanych z danych dostępnych do końca dnia `t`: `Volatility_5d`, `Volatility_20d`, `Mean_Return_5d`, `Mean_Return_20d` i `Mean_Absolute_Return_5d`. Baseline to RSS z poprzednich pięciu zwrotów.

## Protokół

Dane nie są losowo mieszane. Główny podział obejmuje 2187 obserwacji treningowych, pięć pominiętych wierszy odstępu i 549 testowych. Pięciowierszowa przerwa zapobiega nakładaniu się pięciodniowych celów treningowych i zwrotów testowych. `StandardScaler` jest dopasowywany wewnątrz pipeline’u wyłącznie na treningu; Ridge używa `alpha=1.0`. Ujemne prognozy są obcinane do zera. Główną metryką jest MAE.

Dodatkowo projekt wykonuje cztery rozszerzane ewaluacje chronologiczne, ablację `Volatility_20d`, porównanie regresji liniowej z Ridge, sparowany blokowy bootstrap błędów oraz eksperyment z 30 losowymi próbami treningowymi. Okna i sąsiednie cele pochodzą z tej samej serii i nie są niezależnymi próbami. Hipotezy są eksploracyjne; nie wykonano backtestu strategii.

## Środowisko i uruchomienie w Windows PowerShell

Projekt sprawdzono na **CPython 3.14.0**. Bezpośrednie zależności są przypięte w `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\download_data.py
.\.venv\Scripts\python.exe src\explore_data.py
.\.venv\Scripts\python.exe src\analyze_returns.py
.\.venv\Scripts\python.exe src\prepare_features.py
.\.venv\Scripts\python.exe src\train_model.py
.\.venv\Scripts\python.exe src\evaluate_results.py
.\.venv\Scripts\python.exe src\rolling_evaluation.py
.\.venv\Scripts\python.exe src\feature_ablation.py
.\.venv\Scripts\python.exe src\model_comparison.py
.\.venv\Scripts\python.exe src\bootstrap_uncertainty.py
.\.venv\Scripts\python.exe src\validation_comparison.py
.\.venv\Scripts\python.exe src\audit_experiment.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src tests
```

Pobieranie danych wymaga połączenia z internetem. Testy, audyt, przygotowanie cech i modele korzystają z lokalnych plików; nie pobierają danych samodzielnie.

## Skrypty i artefakty

- `download_data.py` zapisuje surowe ceny; `prepare_features.py` tworzy `data/processed/volatility_features.csv`.
- `train_model.py` zapisuje `reports/test_predictions.csv`; `evaluate_results.py` tworzy wykres testu.
- `rolling_evaluation.py` zapisuje `reports/rolling_evaluation.csv`.
- `feature_ablation.py` zapisuje `reports/feature_ablation_main.csv` i `reports/feature_ablation_rolling.csv`.
- `model_comparison.py` zapisuje `reports/model_comparison_main.csv` i `reports/model_comparison_rolling.csv`.
- `bootstrap_uncertainty.py` zapisuje `reports/model_comparison_test_predictions.csv` i `reports/bootstrap_uncertainty.csv`.
- `validation_comparison.py` zapisuje `reports/validation_subset_runs.csv`, `reports/validation_comparison.csv` oraz `reports/figures/validation_comparison.png`.
- `analyze_returns.py` tworzy `reports/figures/price_and_volatility.png`; `evaluate_results.py` tworzy `reports/figures/test_predictions.png`.
- `audit_experiment.py` niezależnie sprawdza cechy, cel i baseline.
- Testy niezmienników znajdują się w `tests/test_research_utils.py`; wspólne ścieżki i obliczenia są w `src/project_paths.py` i `src/research_utils.py`.

## Audyt i ograniczenia

Badanie dotyczy jednego instrumentu i jednej historii z lat 2015–2025. Cele dla sąsiednich dat nakładają się, dodatkowe okna nie są niezależne, a hipotezy powstały w toku eksploracji. Eksperyment losowania próby treningowej wykorzystuje jeden stały okres testowy. Bootstrap opisuje niepewność w jednym teście i zależy od długości bloków. Niższy błąd prognozy nie dowodzi, że strategia inwestycyjna byłaby zyskowna.

Szczegółowa metodologia, wyniki i interpretacja znajdują się w [`reports/research_report.md`](reports/research_report.md). Opis eksperymentu wrażliwości na dobór próby treningowej znajduje się w [`reports/validation_experiment.md`](reports/validation_experiment.md).

## Cel projektu

Projekt ma charakter edukacyjny i badawczy. Jego głównym celem jest testowanie hipotez, kontrola metodologii i uczciwa interpretacja wyników, a nie wyłącznie minimalizacja błędu. Nie stanowi porady inwestycyjnej.
