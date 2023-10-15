# Enkle nevrale nettverk

Last ned all koden enkelt med `git clone https://github.com/Tobias-Opsahl/ai_foredrag.git` dersom du har git-installert,
eller last ned filene manuelt ved å åpne dem og trykke `download` oppe til høyre. Da trenger man ikke å laste ned `README.md`, `.gitignore` eller `generate_data.py`. Pass på å lagre filene i samme mappe. Bruk `cd` i terminalen til å komme frem til riktig mappe i terminalen.

Pass på at `numpy` og `matplotlib` er installert. Om ikke, kjør `pip install numpy` og `pip install matplotlib` i terminalen (eventuelt `pip3` istedenfor `pip`).
Dersom `python` ikke er installert, kan det lastes ned og innstalleres [her](https://www.python.org/downloads/).

## Hvordan kjøre programmene

(Noen må kanskje bytten ut `python` med `python3`, spesielt på litt gamle Linux eller MacOS systemer)

Etter hver kjøring må figuren krysses ut (rødt kryss øvert til høyre eller `ctrl + w` (windows og linux) eller `cmd + w` (mac)).

### Steg 1, gjett på vekter

Kjør `python simple_nn.py`. Du vil bli spurt om å inpute to vekter. Skriv inn ett tall for hver av dem, og se hva resultatet blir. Du kan skrive ingen verdi, altså bare trykke `enter`, for å se en default verdi av vektene.
Det vil bli plottet og vist en graf. Denne inneholder dataene, der klasse en er markert med blå sirkler, og klasse null er markert med rød kryss. Bakgrunnen viser hvilke data modellen klassifiserer som klasse en eller null. Overskriften viser accuracy-en (prosent av data-punkter som ble klassifisert riktig). Prøv å endre vektene slik at accuracyen blir 100%.

Etter hver kjøring må figuren krysses ut (rødt kryss øvert til høyre eller `ctrl + w` (windows og linux) eller `cmd + w` (mac)).

### Steg 2, animer gradient descent

Kjør `python simple_nn.py --run` for å animere hvordan klassifikasjonene ser ut etter hvert steg av gradient descent.
Hver av framsene viser hvordan nettverket klassifiserer dataen etter hver epoch (iterasjon av alle data-punktene).
Det er mulig å endre start-verdiene på vektene og learning-rate, men det må gjøres nederst i programmet.

### Steg 3, animer nevralt nettverk med flere lag

Kjør `python two_layer_nn.py` og input et tall for learning rate (trykk kun `enter` for en god standard verdi). Dette vil animere gradient descent for et nettverk med to lag. Når det er to lag i et nevralt nettverk, er det mer fleksibelt, og kan tilpasse seg ikke-linærheter i dataen.
Prøv så med forskjellige learning rates. Om learning raten er for stor vil du se at klassifikasjonene flytter seg kaotisk uten å konvergere (nærme seg en stabil tilstand), og om den er for lav vil det gå så treigt at ender opp i et dårlig lokalt optimum.

For spørsmål kontakt `tobiasao@uio.no`.

![no gif :(](https://media.giphy.com/media/maNB0qAiRVAty/giphy.gif)
