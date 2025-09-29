# Tworzenie Dużego Modelu Językowego od zera
Praca oparta o książke [Build a Large Language Model](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167) - Sebastian Raschka.

Poniżej znajduje się moje krótkie opracowanie kolejnych rozdziałów książki
# Jak działa LLM?

## Stworzenie słownika  

Słownik to **tabela odwzorowań (lookup table)**, która zawiera tokeny – czyli fragmenty słów – oraz przypisane im liczby (identyfikatory). Dzięki temu możemy zamienić język naturalny na sekwencję liczb. Fragment słownika może wyglądać następująco:  

```json
      "Ġczu": 749,
      "kim": 750,
      "Ġjakie": 751,
      "Ġczłowie": 752,
      "Ġsiebie": 753,
      "Ġ(": 754,
      "szedł": 755,
      "tnie": 756,
      "ran": 757,
      "Ġkto": 758
```

> Symbol `Ġ` oznacza spację. To sposób kodowania informacji o tym, że token pojawia się na początku słowa. Jest to technika zastosowania dla GPT-2  

Jak widać, tokeny często odpowiadają powtarzającym się fragmentom słów. Ten sam token może występować w różnych słowach, na przykład:  

- `ran`-`kiem`
- `po`-`ran`‑`na`  

W takich przypadkach token `ran` (o ID `757`) będzie użyty w słowach takich jak *rankiem*, *porankiem*, *rano* itd.  

W ten sposób całe zdanie, np. `Oto fragment przykładowego wejścia`, może zostać przedstawione jako sekwencja liczb odpowiadających tokenom, np. `3014, 7020, 2324, 1679, 3338, 735`  

Dla ilustracji proces podziału może wyglądać tak:  

```
Oto -> fragment
Oto fragment -> przykład
Oto fragment przykład -> owego
Oto fragment przykładowego -> wej
Oto fragment przykładowego wej -> ścia
```

## Embedding

Następnie tokeny zamienia się na wektory liczbowe w przestrzeni wielowymiarowej. Na początku (przed trenowaniem) wartości tych wektorów są losowe. Podczas treningu, dzięki propagacji wstecznej, wektory ulegają optymalizacji. Z czasem słowa używane w podobnych kontekstach (np. *kot*, *pies*) dostają wektory, które znajdują się bliżej siebie w tej przestrzeni.

Przykład: token `kot` może odpowiadać numerowi `120` w słowniku, a jego wektor początkowy wyglądać np. tak: `[1.2, 5.3]`.

### Embedding pozycyjny

Problem: sam embedding tokenu nie niesie informacji o jego pozycji w zdaniu. Dlatego do wektorów tokenów dodajemy wektory pozycyjne (tzw. *positional embeddings*), które działają jak *offset* zależny od tego, na którym miejscu (indeksie) w zdaniu jest dany token.

**Wracając do przykładu:**

Reprezentacja ciągu `kot kot kot` to najpierw

```
[1.2, 5.3], [1.2, 5.3], [1.2, 5.3]. <- brak informacji o pozycji
```


Następnie dodajemy wektory pozycji, np. `[1.1, 1.2], [2.1, 2.2], [3.1, 3.2]`.

```
  [1.2, 5.3], [1.2, 5.3], [1.2, 5.3]
+ [1.1, 1.2], [2.1, 2.2], [3.1, 3.2]
------------------------------------
= [2.3, 6.5], [3.3, 7.5], [4.3, 8.5]
```

Otrzymując wektor końcowy: `[2.3, 6.5], [3.3, 7.5], [4.3, 8.5]`

Dzięki temu w ostatecznym wektorze mamy zakodowaną zarówno informację o tym, **jaki** to token, jak i o tym, **gdzie** występuje w zdaniu. To z kolei pozwala kolejnym warstwom modelu analizować tokeny względem siebie w kontekście całej sekwencji. 

