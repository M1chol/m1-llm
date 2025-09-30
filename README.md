# Tworzenie Dużego Modelu Językowego
Praca oparta o książke [Build a Large Language Model - Sebastian Raschka](https://www.amazon.com/Build-Large-Language-Model-Scratch/dp/1633437167). Całość jest dostępna w również w [PDF](https://vlanc-lab.github.io/mu-nlp-course/teachings/Build_a_Large_Language_Model_(From_Scrat.pdf).)

Poniżej znajduje się moje krótkie opracowanie kolejnych rozdziałów książki
> [!NOTE]
> Opracowanie rozdziały: 1, 2
> 
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

## Osadzenie (ang. embedding)

Następnie tokeny zamienia się na wektory liczbowe w przestrzeni wielowymiarowej. Na początku (przed trenowaniem) wartości tych wektorów są losowe. Podczas treningu, dzięki propagacji wstecznej, wektory ulegają optymalizacji. Z czasem słowa używane w podobnych kontekstach (np. *kot*, *pies*) dostają wektory, które znajdują się bliżej siebie w tej przestrzeni.

Przykład: token `kot` może odpowiadać numerowi `120` w słowniku, a jego wektor początkowy wyglądać np. tak: `[1.2, 5.3]`.

### Osadzenie pozycyjne (ang. positional embedding)

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

## Mechanizmy samouwagi (ang. self-attention)
TODO

![Image](images/mechanizmy-uwagi.png)


Dla czego dzielimy przez pierwiastek z d_k?? Po to aby uniknąć małych gradientów. Jeżeli nie podzielimy przez pierwiastek z d_k to małe gradienty będą wpływały (w bardzo nieznaczny sposób) na uczenia bardzo je spowalniając

**Do przeczytania**

https://transformer-circuits.pub/2021/framework/index.html

**Do obejrzenia**

https://www.youtube.com/watch?v=kCc8FmEb1nY

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

https://www.youtube.com/watch?v=OFS90-FX6pg