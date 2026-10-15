# Prognozowanie zmienności SPY z wykorzystaniem uczenia maszynowego

## 1. Cel projektu

W tym projekcie chciałem sprawdzić, czy prosty model uczenia maszynowego może lepiej przewidywać krótkoterminową zmienność rynku niż metoda oparta na ostatnich zmianach cen.

Do eksperymentu wykorzystałem dane ETF-u SPY, który odwzorowuje zachowanie indeksu S&P 500. Zamiast próbować przewidzieć, czy cena wzrośnie, czy spadnie, skupiłem się na oszacowaniu wielkości ruchów cen w ciągu kolejnych pięciu sesji giełdowych.

Do porównania wybrałem regresję Ridge oraz prostą metodę bazową wykorzystującą dane z poprzednich sesji.

## 2. Dane

Dane pobrałem z Yahoo Finance przy użyciu biblioteki `yfinance`. Obejmują okres od stycznia 2015 do grudnia 2025 roku.

Początkowy zbiór zawierał 2766 obserwacji. Po przygotowaniu cech i usunięciu wierszy, dla których nie można było obliczyć wszystkich wymaganych wartości, pozostało 2741 obserwacji.

Sprawdziłem również jakość danych. Daty były uporządkowane, nie występowały duplikaty dat ani brakujące wartości w przygotowanym zbiorze.

## 3. Co model ma przewidywać?

Celem modelu jest oszacowanie zmienności z kolejnych pięciu sesji giełdowych. W tym celu wykorzystałem sumę kwadratów dziennych stóp zwrotu, a następnie obliczyłem z niej pierwiastek.

Zastosowana definicja celu ma postać:

\[
Y*t = \sqrt{\sum*{i=1}^{5} r\_{t+i}^{2}}
\]

gdzie \(r\) oznacza dzienną stopę zwrotu.

W praktyce oznacza to, że większe zmiany cen mają większy wpływ na wynik. Model nie przewiduje kierunku przyszłej zmiany ceny, tylko wielkość ruchów w nadchodzącym okresie.

## 4. Przygotowanie cech

Do modelu wykorzystałem pięć cech obliczonych na podstawie historycznych stóp zwrotu:

- zmienność z ostatnich 5 sesji,
- zmienność z ostatnich 20 sesji,
- średnią stopę zwrotu z ostatnich 5 sesji,
- średnią stopę zwrotu z ostatnich 20 sesji,
- średnią wartość bezwzględnej stopy zwrotu z ostatnich 5 sesji.

Wszystkie cechy wykorzystują dane dostępne najpóźniej w dniu tworzenia prognozy. Cel odnosi się natomiast do kolejnych pięciu sesji.

Dzięki temu model nie powinien korzystać z przyszłych stóp zwrotu jako danych wejściowych.

## 5. Porównywane metody

### Metoda bazowa

Jako punkt odniesienia wykorzystałem pierwiastek z sumy kwadratów stóp zwrotu z poprzednich pięciu sesji. Jest to prosta metoda, która zakłada, że ostatnia aktywność rynku może pomóc w oszacowaniu najbliższej przyszłej zmienności.

### Regresja Ridge

Drugą metodą jest regresja Ridge, czyli model regresyjny z regularyzacją L2. Przed treningiem zastosowałem `StandardScaler`, aby ujednolicić skalę cech. Skalowanie było dopasowywane wyłącznie na danych treningowych.

W modelu ustawiłem parametr `alpha = 1.0`. Przewidywane wartości ograniczyłem od dołu do zera, ponieważ zmienność nie powinna być ujemna.

## 6. Podział danych i ocena wyników

Dane podzieliłem chronologicznie, a nie losowo. Model uczył się na wcześniejszych obserwacjach, a jego wyniki sprawdzałem na późniejszych danych.

Do treningu wykorzystałem 2187 obserwacji, a do testu 549 obserwacji. Pomiędzy tymi zbiorami zostawiłem pięć obserwacji odstępu, ponieważ cel obejmuje przyszłe pięć sesji.

Jako główną miarę jakości wykorzystałem MAE, czyli średni błąd bezwzględny. Im niższa jego wartość, tym mniejsza jest przeciętna różnica między prognozą a wartością rzeczywistą.

## 7. Główne wyniki

Na głównym zbiorze testowym otrzymałem następujące rezultaty:

| Metoda         |      MAE |
| -------------- | -------: |
| Metoda bazowa  | 0,008005 |
| Regresja Ridge | 0,006306 |

Regresja Ridge osiągnęła o **21,22% niższy MAE** niż metoda bazowa.

Oznacza to, że w tym eksperymencie model lepiej szacował przyszłą zmienność niż prosta metoda wykorzystująca ostatnie sesje. Jest to jednak wynik dla konkretnego instrumentu i okresu historycznego, a nie gwarancja podobnej skuteczności w przyszłości.

## 8. Sprawdzenie modelu na kolejnych okresach

Żeby sprawdzić, czy wynik nie zależy wyłącznie od jednego podziału danych, przeprowadziłem dodatkową ewaluację na czterech kolejnych okresach. W każdym przypadku model był trenowany na wcześniejszych danych, a następnie oceniany na następnym okresie.

| Okres testowy | MAE metody bazowej | MAE Ridge | Poprawa |
| ------------- | -----------------: | --------: | ------: |
| 2021–2022     |           0,008768 |  0,007718 |  11,97% |
| 2022–2023     |           0,007640 |  0,005534 |  27,57% |
| 2023–2024     |           0,006215 |  0,004781 |  23,06% |
| 2024–2025     |           0,009792 |  0,007833 |  20,00% |

Model Ridge osiągnął niższy błąd we wszystkich czterech oknach. Średnia poprawa wyniosła 20,65%, a mediana 21,53%.

Uważam to za istotny element eksperymentu, ponieważ przewaga modelu utrzymała się w kilku różnych okresach. Trzeba jednak pamiętać, że wszystkie okna pochodzą z tej samej serii danych i nie są od siebie całkowicie niezależne.

## 9. Wyniki przy różnym poziomie zmienności

Dodatkowo sprawdziłem, jak obie metody radzą sobie w okresach o różnej przyszłej zmienności.

Podzieliłem obserwacje testowe na dwie grupy: 20% przypadków o najwyższej zrealizowanej przyszłej zmienności oraz pozostałe 80%. Jest to analiza przeprowadzona po uzyskaniu wyników — w momencie tworzenia prognozy przyszła zmienność nie jest jeszcze znana.

| Grupa                    | Liczba obserwacji | MAE metody bazowej | MAE Ridge |
| ------------------------ | ----------------: | -----------------: | --------: |
| Najwyższe 20% zmienności |               110 |           0,017564 |  0,013916 |
| Pozostałe obserwacje     |               439 |           0,005610 |  0,004400 |

W obu grupach Ridge uzyskał niższy błąd. Jednocześnie błędy były znacznie większe w przypadkach o najwyższej zmienności, co pokazuje, że gwałtowne ruchy rynku pozostają trudniejsze do oszacowania.

## 10. Ograniczenia

Projekt ma kilka ograniczeń, które trzeba uwzględnić przy interpretacji wyników.

Po pierwsze, wykorzystałem tylko jeden instrument finansowy — SPY — oraz jeden zakres danych historycznych. Nie sprawdzałem, czy podobne wyniki można uzyskać dla innych aktywów.

Po drugie, cele obejmują kolejne pięć sesji, więc okresy prognoz częściowo się pokrywają. Zastosowałem odstęp między treningiem a testem, aby ograniczyć nakładanie się okresów, ale kolejne okna testowe nadal pochodzą z tej samej serii czasowej.

Po trzecie, testowałem jedną konfigurację regresji Ridge. Nie przeprowadzałem szerokiego przeszukiwania parametrów ani testów istotności statystycznej. Nie mogę więc stwierdzić, że uzyskana poprawa jest statystycznie istotna.

Nie testowałem również strategii inwestycyjnej, kosztów transakcyjnych ani wpływu prognoz na rzeczywiste wyniki finansowe. Niższy błąd przewidywania zmienności nie oznacza automatycznie, że na podstawie modelu można osiągać zyski.

## 11. Podsumowanie

W projekcie przygotowałem dane historyczne SPY, obliczyłem cechy na podstawie wcześniejszych stóp zwrotu i porównałem regresję Ridge z prostą metodą bazową.

Na głównym zbiorze testowym Ridge obniżył MAE o 21,22%. W dodatkowej ewaluacji chronologicznej uzyskał niższy błąd we wszystkich czterech badanych okresach, a średnia poprawa wyniosła 20,65%.

Wyniki sugerują, że w tym eksperymencie regresja Ridge była skuteczniejsza od zastosowanej metody bazowej. Nie oznacza to jednak, że jest najlepszym możliwym modelem ani że jego prognozy można bezpośrednio wykorzystać do podejmowania decyzji inwestycyjnych.

Kolejnym krokiem mogłoby być porównanie innych modeli, wykorzystanie dodatkowych cech i sprawdzenie, czy podobne wyniki pojawiają się na innych instrumentach finansowych.

## 12. Skrypty projektu

Najważniejsze pliki wykorzystane w eksperymencie:

- `src/download_data.py` — pobieranie danych,
- `src/explore_data.py` — wstępna analiza zbioru,
- `src/analyze_returns.py` — analiza stóp zwrotu,
- `src/prepare_features.py` — przygotowanie cech i celu,
- `src/train_model.py` — trening i główna ewaluacja,
- `src/evaluate_results.py` — porównanie prognoz i wykres,
- `src/rolling_evaluation.py` — ewaluacja na kolejnych okresach,
- `src/audit_experiment.py` — niezależne sprawdzenie obliczeń celu i metody bazowej.
