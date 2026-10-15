# Prognozowanie zmienności SPY

Projekt badawczy sprawdzający, czy proste modele regresyjne potrafią prognozować krótkoterminową zmienność ETF-u SPY oraz które decyzje dotyczące konstrukcji modelu mają znaczenie dla wyników.

Zamiast przewidywać kierunek ceny, model szacuje wielkość ruchów cen w kolejnych pięciu sesjach giełdowych. Najważniejszym elementem projektu jest testowanie hipotez i kontrola metodologii, a nie samo uzyskanie jak najniższego błędu.

## Pytania badawcze

1. Czy Ridge poprawia prognozę względem prostej metody bazowej?
2. Czy cecha `Volatility_20d` wnosi użyteczną informację ponad pozostałe cechy?
3. Czy Ridge ma praktyczną przewagę nad zwykłą regresją liniową przy tym samym zestawie cech?

## Najważniejsze wyniki

### Ridge a metoda bazowa

| Metoda         |      MAE |
| -------------- | -------: |
| Metoda bazowa  | 0.008005 |
| Ridge — 5 cech | 0.006306 |

Na głównym zbiorze testowym Ridge uzyskał MAE niższy o **21,22%** względem metody bazowej.

W czterech chronologicznych oknach testowych Ridge również uzyskał niższy MAE. Średnia poprawa wyniosła **20,65%**. Okna pochodzą jednak z tej samej historycznej serii i częściowo pokrywają się z głównym okresem testowym, więc nie są niezależnymi eksperymentami.

### Ablacja cechy `Volatility_20d`

| Wariant                    |      MAE |
| -------------------------- | -------: |
| Ridge — wszystkie cechy    | 0.006306 |
| Ridge bez `Volatility_20d` | 0.006576 |

Po usunięciu `Volatility_20d` MAE wzrósł o **4,28%** na głównym teście. Błąd wzrósł również we wszystkich czterech dodatkowych oknach chronologicznych. Wyniki wspierają hipotezę, że ta cecha jest przydatna w tym modelu i na tym zbiorze danych; nie dowodzą związku przyczynowego ani uniwersalnej użyteczności tej cechy.

### Ridge a zwykła regresja liniowa

| Model               |         MAE |
| ------------------- | ----------: |
| Regresja liniowa    | 0.006306482 |
| Ridge (`alpha=1.0`) | 0.006306344 |

Różnica na głównym teście jest bardzo mała — przewaga Ridge wynosi około **0,00218%**. Ridge wygrywa w trzech z czterech dodatkowych okien, ale różnice również są minimalne.

Bootstrap blokowy dla różnicy MAE (regresja liniowa minus Ridge) dał 95-procentowe przedziały dla względnej poprawy obejmujące zero przy długościach bloku 5, 10 i 20. Nie ma więc przekonujących dowodów na praktyczną przewagę Ridge w tym porównaniu. Nie jest to jednak formalny dowód równoważności modeli.

## Dane i definicja celu

- Instrument: SPY.
- Źródło: Yahoo Finance (`yfinance`).
- Zakres: 2015–2025.
- Początkowo: 2766 obserwacji.
- Po przygotowaniu cech: 2741 obserwacji.

Cel prognozy to pierwiastek z sumy kwadratów dziennych stóp zwrotu z kolejnych pięciu sesji:

\[
Y*t = \sqrt{\sum*{i=1}^{5} r\_{t+i}^{2}}
\]

Jest to miara łącznej wielkości ruchów cen w przyszłym pięciosesyjnym okresie — nie prognoza kierunku ceny ani klasyczne odchylenie standardowe.

Cechy wejściowe:

- `Volatility_5d`
- `Volatility_20d`
- `Mean_Return_5d`
- `Mean_Return_20d`
- `Mean_Absolute_Return_5d`

Dane dzielone są chronologicznie. Skalowanie dopasowywane jest tylko na danych treningowych, a pomiędzy treningiem i testem pozostawiono pięć obserwacji odstępu ze względu na nakładanie się horyzontów celu.

## Struktura projektu

```text
ml_volatility_forecasting/
├── reports/
│   ├── figures/
│   ├── research_report.md
│   ├── rolling_evaluation.csv
│   ├── test_predictions.csv
│   ├── feature_ablation_main.csv
│   ├── feature_ablation_rolling.csv
│   ├── model_comparison_main.csv
│   ├── model_comparison_rolling.csv
│   └── bootstrap_uncertainty.csv
├── src/
│   ├── download_data.py
│   ├── explore_data.py
│   ├── analyze_returns.py
│   ├── prepare_features.py
│   ├── train_model.py
│   ├── evaluate_results.py
│   ├── rolling_evaluation.py
│   ├── audit_experiment.py
│   ├── feature_ablation.py
│   ├── model_comparison.py
│   └── bootstrap_uncertainty.py
├── .gitignore
├── README.md
└── requirements.txt
```

Katalogi `data/raw/` i `data/processed/` są lokalnymi artefaktami i są ignorowane przez Git. Dane można odtworzyć skryptem pobierającym.

## Uruchomienie

Wymagany Python z bibliotekami wymienionymi w `requirements.txt`.

```bash
pip install -r requirements.txt
python src/download_data.py
python src/prepare_features.py
python src/train_model.py
python src/evaluate_results.py
python src/rolling_evaluation.py
python src/audit_experiment.py
python src/feature_ablation.py
python src/model_comparison.py
python src/bootstrap_uncertainty.py
```

Skrypty należy uruchamiać z głównego katalogu repozytorium. Eksperymenty zapisują wyniki do katalogu `reports/`.

## Audyt i ograniczenia

`src/audit_experiment.py` niezależnie odtwarza obliczenia celu i metody bazowej. Audyt wykazał 2741 obserwacji, brak zduplikowanych dat, uporządkowane daty, brak brakujących wartości i zerowy błąd rekonstrukcji celu oraz baseline'u.

Najważniejsze ograniczenia:

- badany jest jeden instrument i jeden zakres historyczny;
- pięciosesyjne horyzonty celu częściowo się pokrywają, co ma znaczenie przy ocenie niepewności;
- dodatkowe okna nie stanowią niezależnych prób;
- analiza grup o najwyższej przyszłej zmienności jest diagnostyką post hoc — przyszła zmienność nie jest znana w momencie prognozy;
- nie przeprowadzono testu strategii inwestycyjnej; dokładniejsza prognoza zmienności sama w sobie nie oznacza zysków;
- hipotezy mają charakter eksploracyjny, a wyniki nie powinny być interpretowane jako gwarancja skuteczności na przyszłych danych.

## Raport badawczy

Pełny opis hipotez, metodologii, wyników i ograniczeń znajduje się w [`reports/research_report.md`](reports/research_report.md).
