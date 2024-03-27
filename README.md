# jorgX_poc

## Opis
Projekt "jorgX_poc" to Proof of Concept (PoC) dla systemu analizy sentymentu wykorzystującego model przetwarzania języka naturalnego (NLP) w TensorFlow i interfejs API TensorFlow w Javie. Projekt składa się z dwóch głównych części: skryptu Pythona do trenowania modelu NLP na danych IMDB oraz aplikacji Java, która wykorzystuje wytrenowany model do przewidywania sentymentu na podstawie recenzji wprowadzonych przez użytkownika.

## Struktura projektu
- `src/main/java/org/example/Main.java`: Główna klasa aplikacji Java, która ładuje wytrenowany model TensorFlow i przeprowadza inferencję na podstawie wprowadzonych recenzji filmów.
- `src/main/resources/model`: Katalog zawierający model TensorFlow. Model musi zostać wytrenowany i zapisany w `src/main/resources/model/saved_model` przy użyciu dołączonego skryptu Pythona.
- `src/main/resources/model/saved_model`: Lokalizacja, w której należy zapisać wytrenowany model TensorFlow (model musi zostać zapisany ręcznie przy użyciu skryptu Pythona ze względu na ograniczenia rozmiaru pliku na GitHubie).

## Wymagania
- Java (wersja 17 lub nowsza)
- TensorFlow dla Javy
- Python (wersja 3.6 lub nowsza)
- TensorFlow (wersja 2.x)
- TensorFlow Datasets

## Jak uruchomić
1. **Trenowanie modelu:**
   - Przejść do katalogu zawierającego skrypt Pythona.
   - Skrypt Pythona można uruchomić lokalnie lub w Google Colab, aby wytrenować model na danych IMDB i zapisać go w `src/main/resources/model/saved_model`.

2. **Uruchomienie aplikacji Java:**
   - Upewnić się, że środowisko Java jest skonfigurowane, a wszystkie wymagane zależności są dostępne.
   - Uruchomić `Main.java` w środowisku IDE lub z linii komend.

## Funkcjonalność
- **Trenowanie modelu:** Skrypt Pythona trenuje model sieci neuronowej do analizy sentymentu, wykorzystując recenzje filmów z bazy danych IMDB.
- **Przewidywanie sentymentu:** Aplikacja Java ładuje wytrenowany model i umożliwia użytkownikowi wprowadzenie recenzji filmu, na podstawie której przewidywany jest sentyment (pozytywny lub negatywny).
