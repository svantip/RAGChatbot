# Dokumentacija za RAG Chatbot za pomoć pri upisu na studije

### Svan Tipurić, Deni Koraca

# Upute za pokretanje aplikacije

```

python3 venv venv

source venv/bin/activate

pip install -r requirements.txt

docker compose up -d

python3 vectorize_and_save_data.py

streamlit run init_chat_model.py

# Aplikacija je dostupna na http://localhost:8501
```

# Dokumenti

Dokumenti koje koristi chatbot preuzeti su sa stranice faksa FIDIT. Ukupno podaci od chatbota se sastoje od 4 datoteke(3 .pdf i 1 .docx).

# Baza podataka

Za vektorsku bazu podataka koristili smo Redis. Redis je baza podataka koja omogućuje brzo i efikasno pretraživanje sličnosti među vektorima i dobro korelira
s frameworkom Langchain koji je korišten u cijelom projektu.

# DeepSeek-R1:8B

Model DeepSeek-R1:8B odabran je zbog svojih naprednih mogućnosti obrade prirodnog jezika. Riječ je o jezičnom modelu s 8 milijardi parametara koji postiže visoke rezultate u zadacima razumijevanja i generiranja teksta.

U kontekstu projekta koristi se kao chatbot model sposoban je generirati koherentne i informativne odgovore, čime se omogućuje prirodan i smislen dijalog s korisnikom.

Ovaj je model prepoznat kao optimalno rješenje za sustave koji zahtijevaju kombinaciju razumijevanja jezika i dohvaćanja informacija temeljenog na značenju.

Jedan od negativnih aspekata ovoga modela u chatbot aplikaciji je da je to thinking model, samim time korisnik aplikacije mora dulje čekati odgovor ali odgovor je kvalitetnije promišljen.

# Podjela teskta na chunk-ove i overlap

Chunk - dio teskta koji je pretvoren u embedding
Overlap - preklapanje dva susjedna chunk-a

Kao što sam ranije napomenuo kroz cijeli projekt korištena je biblioteka Langchain, te su se pomoću njenjih napradnih mogućnosti datoteke parsirane i podjeljene u chunk-ove.
Tekst je podjeljen u 1000 chunk-ova od kojih je 200 overlap postižući idealan omjer chunk-ova i overlap-a u chatbot aplikacijama ovisno o datotekama koje vektorizira.

# Vektorsko pretraživanje Redis-a

Kada korisnik unese pitanje ono se pretvara u embedding i provodi se vektorsko pretraživanje temeljeno na metrikama udaljenosti (cosine sličnost), koje vraća k-najbližih embeddinga prema korisničkom upitu.

# LLM prompt i temperatura

LLM prompt složen je tako da jasno definira ulogu modela kao asistenta za upise na fakultet. U promptu se specificira da model mora odgovarati na temelju prethodno vektoriziranih i parsiranih podataka iz natječaja, bez izmišljanja informacija. Također se određuju smjernice za stil odgovora – odgovori moraju biti jezgroviti, informativni i jezično prilagođeni korisniku (npr. hrvatski jezik ako korisnik piše na srpskom).

Temperatura modela postavljena je na 0.3, čime se smanjuje razina slučajnosti u generiranju odgovora. Niža temperatura znači da će model generirati konzistentnije i predvidljivije odgovore, što je poželjno u sustavu koji pruža točne i pouzdane informacije vezane uz upise na visokoškolske ustanove.

# Korisničko sučelje

Korisničko sučelje izrađeno je pomoću biblioteke Streamlit, koja omogućuje jednostavno kreiranje interaktivnih web-aplikacija u Pythonu. Sučelje omogućuje korisniku da unese pitanje vezano uz upis na studij, prikaže prethodnu komunikaciju, te na intuitivan način prikaže generirani odgovor modela.

Pri svakom unosu:
• Korisnik unosi pitanje putem chat inputa.
• Odgovori se prikazuju u obliku poruka u chat formatu (korisnik / asistent).
• Tijekom generiranja odgovora prikazuje se animacija “Thinking…” radi boljeg korisničkog doživljaja.

Ovakav način interakcije omogućuje prirodan tijek razgovora, nalik komunikaciji sa stvarnim asistentom, čime se poboljšava korisničko iskustvo i razumijevanje informacija o upisima.

<img width="1268" alt="upload_3cebac28c8ea18f106d65862e2a62c93" src="https://github.com/user-attachments/assets/69ceedc1-f9dc-4788-92da-c8df8f58cc3e" />
> Slika korisničkog sučelja

# Arhitektura aplikacije

Aplikacija je modularno organizirana i sastoji se od četiri ključne komponente koje međusobno surađuju:

1. Korisnički sloj (Streamlit UI)

- Prikuplja korisničke upite i prikazuje odgovore.
- Omogućuje vođenje dijaloga s korisnikom kroz chat sučelje.

2. LangGraph workflow

- Implementira logiku toka razgovora koristeći StateGraph.
- Svaki korisnički unos prolazi kroz definirani tok, koji obuhvaća dohvat dokumenata i poziv LLM-a.
- MemorySaver omogućuje spremanje konteksta razgovora kako bi odgovori bili konzistentni.

3. Vektorska baza podataka (Redis)

- Pokrenuta unutar dockera sa `docker-compose up`
- Sprema embeddinge dokumenata dobivenih iz parsiranih .pdf i .docx datoteka.
- Koristi se za brzo semantičko pretraživanje najrelevantnijih dokumenata u odnosu na korisnički upit.
- Povezana je s aplikacijom preko langchain_community.vectorstores.Redis.

5. LLM komponenta (DeepSeek-R1:8B)

- Generira tekstualne odgovore na temelju prompta i konteksta embeddinga.
- Prompt jasno definira ulogu modela i stil komunikacije (jezgrovito, informativno, bez izmišljanja podataka).

Komunikacija među komponentama odvija se u realnom vremenu. Cijeli sustav je dizajniran tako da omogućuje fleksibilnu nadogradnju, primjerice dodavanje novih izvora podataka, autentifikaciju korisnika, ili proširenje na druge obrazovne institucije.

![upload_06ba5988c80b4d5f97349222a2e482fb](https://github.com/user-attachments/assets/48c33ac7-82c0-4f8b-9bd4-b9770365ae69)

> Primjer toka generiranja odgovora

# Chatbot

Chatbot je implementiran korištenjem RAG arhitekture (Retrieval-Augmented Generation), s lokalno pokrenutim LLM-om (DeepSeek-R1:8B) i vektorskom bazom podataka Redis pokrenutom unutar Docker kontejnera.

Dodatne funkcionalnosti sustava uključuju:
• Multijezičnost – omogućena zahvaljujući sposobnostima velikog jezičnog modela da prepoznaje i generira tekst na više jezika.
• Interaktivno korisničko sučelje – izrađeno pomoću Streamlit-a, pruža intuitivnu komunikaciju korisnika s chatbotom u stvarnom vremenu.
• Memorija razgovora – implementirana pomoću LangGraph-a i MemorySaver mehanizma, što omogućuje vođenje višekratnih interakcija bez gubitka konteksta.

## Evaluacija chatbota

U nastavku se nalazi evaluacija chatbotovih odgovora sa pitanjem, odgovorom i ocjenom svakog prompta.

### 1. Koje dokumente trebam predati za upis na preddiplomski studij?

Odgovor chatbota:

```
<think> Hmm, korisnik je pitao o dokumentima za upis na preddiplomski studij. U kontekstu koji mi je bio dostavljen, vidim da se govori o sveučilišnom integriranom prijediplomskom i diplomskom studiju te razlikovnome programu za stjecanje nedostajućih znanja.
Naravno, trebala bih odmah prepoznati da je korisnik u pitanju upisa na preddiplomski studij, a ne na magistrat ili drugi stupanj. To je ključna razlika jer zahtjevi za dokumentima mogu biti drugačiji ovisno o kategoriji studija.

Iz konteksta vidim da se korisnik interesuje za upis na preddiplomski sveučilišni studij. To podrazumijeva nekoliko važnih informacija koje treba odrediti: uvjete ulaznice, potrebne dokumentacije i postupak dostave.

Korisnik vjerojatno želi znati što mora pripremiti za upis. To uključuje standardnu fotografiju za dokumente te dokaze o uplatama za upisninu i školarinu, ako se primarno bavio razlikovnim postupkom.

Kako je korisnik pao na hrvatski (prema pravilima prometa), moj odgovor će biti na hrvatskom. Treba objasniti sve važne dokumente i upozoriti na potrebnu preciznost, posebno ako ima inozemne obrazovanje.

Također ću uključiti informaciju o tome gdje se nalaze detaljnije upute - na web stranicama PFRI-a. To je korisno jer korisnik može pronaći više informacija ako želi. </think> Kandidatima za preddiplomski studij (razlikovni program) potrebno je predati sljedeće dokumente:

Upisni broj s rang liste: Ovo je ključni dokument koji se dobije iz postupka prijave na natječaj.
Fotografiju: Standardna fotografija za dokumente.
Dodatno (ako se primarno bavite razlikovnim programom):

Dokaz o uplati upisnine: Iznos je 60,50 EUR i plaća se na navedeni IBAN.
Dokaz o uplati školarine: Ako ste u kategoriji "izvanredni status" ili "stranac", morate uplatiti i školarinu (1.100,00 EUR) prema navedenom IBAN-u.
Informacije su dostupne na:

www.postani-student.hr
Web stranici Fakulteta: https://www.pfri.uniri.hr/web/hr/ES-upisi_preddiplomski.php
Molimo pazite na točnost upisnog broja i dostave svih potrebnih dokumenata prema uputama fakulteta.
```

Chatbot daje relevantan odgovor na postavljeno pitanje. (4/5)

### 2. Koliko je školarina za diplomski studij informatike?

Odgovor chatbota:

```
<think> Hmm, the user is asking about tuition fees for a master's degree in Informatics. Let me check the context provided.
Okay, looking at the context... It mentions that for doctoral studies (which come after a master's), there are semester fees and total maximum expenses of €607 per semester for up to six semesters. But it doesn't say anything about master's tuition fees specifically.

The user might be confused between different levels of study - perhaps they're asking about the master's degree but saw information about doctoral studies in the context. Or maybe they're referring to a different program not mentioned here.

Since I don't have specific information about master's tuition fees, and my instructions say to only answer if I find the information or say "I don't know", I should be honest here. The most accurate response is that this information isn't available in the provided context.

The user seems to be an international student interested in studying at this faculty, possibly from a technical field given their question about Informatics fees. They're being practical by asking about costs directly - smart move for planning purposes!

I should keep my answer short and clear as instructed, just stating that I don't have the information while making it clear what level of study was mentioned in the context versus what they're actually asking. </think> Nemam informacije o upisnoj za diplomski studij Informatike.
```

Chatbot iako ima informacije, nije ju pronašao i daje odgovor da nema tu informaciju. (1/5)

### 3. Koje Dokumente moram priložiti kako bih upisao preddiplomski studij?

Odgovor chatbota:

```
<think> Hmm, korisnik pita o dokumentima potrebnim za prijavu na preddiplomski studij. Na temelju konteksta koji mi je dostavljen vidim da se radi o sveučilišnom integriranom prijediplomskom i diplomskom studiju s 300 ECTS bodova u području ekonomije.
Korisnik vjerojatno želi upisati preddiplomski studij, ali nije siguran koji dokumenti su potrebni. To je razumna pitanja budući da postupak prijave zahtjeva određene dokumente kako bi se provjerilo ispunjenje uvjeta.

Iz konteksta vidim da za preddiplomski studij pedagogije ili predškolski stručni studij potrebno je 180 ECTS bodova. Također piše da kandidati sa stručnih studija moraju izvršiti program za stjecanje nedostajućih znanja prije nego što upisu diplomski studij.

Ali korisnik pita specifično o dokumentima za preddiplomski studij, pa ću odgovorit na temelju informacija koje sam dobila. Iz konteksta vidim da se prijave za razredbeni postupak podnose na posebnom obrascu koji se može preuzeti sa web stranice fakulteta.

Korisnik je vjerojatno nervozan i želi bitno uputstvo o tome što treba pripremiti. Moram bit jasan i točan u odgovoru, jer dokumentacija za prijavu na studij može bit ključna za njihov uspjeh.

Naravno, moram pazit da ne izmišljam informacije ako ih nema u kontekstu. Ovdje imam dovoljno informacija o tome koji dokumenti su potrebni - glavni je obrazovanje i prilog sa podatcima za program za stjecanje nedostajućih znanja.

Također vidim da rok za prijavu je 25. rujna 2025., pa bi korisnik bio informatan o tome što ne treba propustit. </think> Kada želite upisati preddiplomski studij, moratete priložiti dokumente koji potvrđuju vaše obrazovanje i ispunjenost uvjeta. Na temelju konteksta, za prijavu na razredbeni postupak (preddiplomski studij) potrebni su:

Obrazovanje:

Potvrda o završetku sveučilišnog preddiplomskog studija ili stručnog studija s odgovarajućim ECTS bodovima (prema uvjetima za vaš tip kandidata).
Program za stjecanje nedostajućih znanja:

Potvrda o polaganju i završetku programa za stjecanje nedostajućih znanja, vještina i kompetencija (ako ste kandidati sa stručnog studija).
Prijave se podnose na posebnom obrascu koji možete preuzeti na mrežnim stranici fakulteta.

Važno: Za pristup razredbenom postupku (B) nužno je izvršiti program za stjecanje nedostajućih znanja prije nego što upitate diplomski studij.
```

Chatbot daje relevantni odgovor no pojmovi preddiplomski i diplomski studij su mu preslični pa pronalazi krivi odgovor. (4/5)

### 4. Do kada moram poslati prijave za preddiplomski studij?

Odgovor chatbota:

```
<think> Hmm, korisnik je upitao do kad mora poslati prijave za preddiplomski studij na Građevinski fakultet u Rijeci. Tačno, u kontekstu piše da se prijave za razredbeni postupak primaju online do 23. lipnja 2025.
Moram paziti da odgovorim tačno i korisno. Prvo je važno da iskoristim informaciju iz konteksta - datum je jasan naveden, ali moram biti pažljiv jer postoji mogućnost da neko zavreli preddiplomski studij ima dodatne zahteve.

Korisnik verovatno želi znati tačan rok za prijavu kako bi bio spreman ranije. Možda je već radio na dokumentaciji ili predmetima priprema, pa je ovo ključna informacija za njegov planiranje.

Još jedno: u kontekstu piše i da se provjera znanja odvija 27. lipnja, ali to nije direktno vezano za pitanje korisnika. Ipak, biće dobro da ga podsjetim i na to kako ne bi bio iznenaden kad godar.

Odgovorit ću na hrvatskom jeziku jer korisnik pita na srpskom. Bit ću kratkostručan i jasan, samo sa znanjem iz konteksta, bez dodatnih informacija koje nisu potrebne. </think> Prijave za razredbeni postupak mogu se prijaviti do 23. lipnja 2025. online.
```

Chatbot kratko i jasno odgovara na postavljeno pitanje. (5/5)

### 5. Koje su tvoje mogućnosti kao asistentu?

Odgovor chatbota:

```
<think> Hmm, korisnik je pitao o mojim mogućnostima kao AI asistenta. To je opšta pitanja koja se često postavlja prilikom upoznavanja nove tehnologije.
Naravno, moram odgovoriti na srpskom jeziku jer korisnik nije specificirao drugi jezik, a prema mojim pravilima bih trebao koristiti informacije iz konteksta. Ali ovo pitanje nije direktno vezano za kontekst koji mi je bio predstavljen - to je bila upravo tačka o upisima na fakultet.

Moguće je da korisnik zamišlja da možda želim da mu pomognem sa spiskom za upis, ali nije specificirao. Ipak, najbolje je da budem razumljiv i odgovorim na to pitanje bez pretpostavke.

Korisnik vjerojatno želi znati do koje mjere mogu biti pouzdan u meni kao AI asistentu. Možda se boji da ću dati nepotrebne informacije ili da nisam dovoljan za određene zadatke.

Moj odgovor treba biti jasan i jezikovno neutralan, ali pošto korisnik nije specificirao jezik, a u kontekstu piše na hrvatskom, odlučio sam da odgovorim na srpskom kako bi bio prikladan.

Napravio sam pregled mogućih funkcija: prevođenje, objašnjenja informacija iz konteksta i podršku u razumijevanju pravila upisa. To pokazuje moje ključne sposobnosti bez previše tehničkih detalja.

Korisnik je vjerojatno student ili budući student, pa je važno da mu objasnim što mogu učiniti konkretne i praktične informacije. Naglasio sam granice - samo koristim informacije iz konteksta i odgovaraj samo ako ih ima.

Ovo je dobro prilagođen odgovor jer pokazuje korisnosti AI asistentov bez previše komplikacija. </think> Moje mogućnosti su ograničene na:

Prevođenje: Možete me pitati da prevedem tekst iz hrvatskog jezika na srpski ili obratno, uzimajući u obzir jednostavne konverzacije.
Objašnjenje informacija iz konteksta: Koristeći samo podatke iz trenutno prikazanog kontekста (koji se tiče upisa na fakultet), možete me pitati da vam objasnim postavljena pitanja i pravila vezana za upis.
Podrška u razumijevanju: Možete me koristiti za jasnije objašnjenje informacija koje su već navedene u kontekstu.
Važno je da napomenem da samo koristim informacije iz trenutnog prikazanog konteksta i odgovaraj samo ako nađem tačan odgovor.
```

Chatbot jasno objašnjava korisniku svoju svrhu, mogućnosti i limitacije. (5/5)

# Zaključak

Iako potpuno precizna i kontekstualno bogata pretraga dokumenata još nije u potpunosti ostvarena zbog nedostataka relevantnih dokumenata, projekt predstavlja snažnu osnovu za daljnji razvoj. Sustav već sada demonstrira ključne principe RAG arhitekture, a postavljena je i modularna mikroservisna infrastruktura koja omogućuje skalabilnost i fleksibilnost.

Daljnjim unapređenjem osobito kroz finetuning modela, proširenje baze dokumenata i optimizaciju upita moguće je značajno povećati točnost odgovora i korisničko iskustvo. Ovaj pristup otvara vrata razvoju inteligentnog asistenta prilagođenog obrazovnim sustavima i specifičnim potrebama krajnjih korisnika.

# Literatura

https://python.langchain.com/docs/introduction/
https://redis.io/about/about-stack/
https://medium.com/credera-engineering/build-a-simple-rag-chatbot-with-langchain-b96b233e1b2a
