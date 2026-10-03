### Aktuell: begrenzter Experten-Vorlauf

Der Reader lädt jetzt standardmäßig den nächsten **bereits ausgewählten** Run
vor, bevor die Callbacks des aktuellen Runs rechnen. Die Linux-I/O kann dadurch
mit der Berechnung überlappen. Kein Konstantencache, kein neuer Workerpool und
keine dauerhafte Gewichtsretention wurden hinzugefügt; die 128-KiB-Häppchen bleiben.
Pro Aufruf sind höchstens zwei transiente Runs aktiv (gewöhnlich zusammen bis
64 MiB, ausgenommen einzelne größere Expertenmatrizen und Seitenausrichtung).
Abbruch und Panic geben auch den vorgezogenen Run frei.

Intel Core i7-4790T, 15,5 GiB RAM, Go 1.27.1/SIMD, vorhandenes GPT-OSS-120B,
64-MiB-Fenster. Je Variante ein Aufwärmlauf, danach wechselnde Reihenfolge;
je Lauf frischer Reader, Engine und KV-Zustand. Gemessen wird `Engine.Generate`
mit `context.Background()`, einschließlich Textdekodierung, ohne Initialisierung,
Prompt-Encoding und Terminalausgabe. OS-Caches wurden nicht geleert und die
CPU-Frequenz nicht fixiert. Die Serien dürfen nicht direkt mit früheren absoluten
Zeiten verglichen werden, sondern jeweils mit ihrem eigenen Ausgangspfad.

| Szenario | Threads | Messpaare | Ohne Vorlauf, Median (Bereich) | Mit Vorlauf, Median (Bereich) | Verkürzung |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Hallo, sag nur Ja!`, 15 Prompt-Tokens, Limit 5 | 8 | 3 | 30,145 s (28,867–30,229) | 27,441 s (26,986–28,068) | 8,97 % |
| `Zähle von eins bis zehn, mit Kommas getrennt.`, 23 Prompt-Tokens, Limit 8 | 8 | 2 | 55,697 s (55,579–55,816) | 50,935 s (50,569–51,300) | 8,55 % |
| Kurzer Prompt wie oben | 4 | 3 | 32,732 s (32,704–32,740) | 29,161 s (29,082–29,203) | 10,91 % |

Tokenfolgen, Texte, Endaktivierungen und Hashes der belegten KV-Positionen waren
in sämtlichen Läufen identisch. Der kurze Prompt erzeugte `Ja.` (`30109`, `13`),
der zweite `1, ... ... ... ... ... ...` (acht Tokens); die Antwortqualität wurde
nicht verändert. Der Gewinn entsteht überwiegend beim Prefill. Insbesondere die
erste Serie zeigt keinen konsistenten Decode-Gewinn; sehr lange Generierungen
und andere Hardware sind nicht abgesichert. Ein allgemeiner Gewinn von 10 %
wird daher **nicht** behauptet.

Der abschließende Vier-Thread-Vergleich prüft zusätzlich identische ausgewählte
Bytes, Mapping- und Prefetch-Zahlen, null Prefetch-Fehler und ausgeschaltete
Gewichtsretention. Pro Lauf bleiben etwa 23,6 Millionen 512-Byte-Eingabeblöcke:
das ist kein rein RAM-warmer Benchmark. Lokale Rohprotokolle:
`bin/lookahead-generate-sync.log`, `bin/lookahead-generate-long.log`,
`bin/lookahead-generate-4threads.log`.

Abschließender CLI-Smoke-Test mit acht Threads, kurzem Prompt und `-expert-stats`:
`Ja.` in 26,844 s, 114.600 begrenzte Prefetch-Anfragen, null Fehler und null
gesperrte Gewichtsbytes (`bin/lookahead-cli.log`). Dieser einzelne Lauf ist kein
zusätzlicher A/B-Nachweis.

### Aktuellen Vergleich wiederholen

```bash
GOEXPERIMENT=simd STREAM_PT_GENERATE_PERF=1 STREAM_PT_WORKERS=8 \
STREAM_PT_PERF_REPEATS=3 \
go test ./tests -run '^TestGenerateModelPerformance$' -count=1 -v -timeout=10m

GOEXPERIMENT=simd go run . -t 8 -w 64 -n 5 -p 'Hallo, sag nur Ja!'
GOEXPERIMENT=simd go run . -t 8 -w 64 -n 5 -p 'Hallo, sag nur Ja!' -no-expert-lookahead
```

`STREAM_PT_PERF_PROMPT` und `STREAM_PT_PERF_TOKENS` variieren den Workload.
`STREAM_PT_PERF_REQUIRE_GAIN=1` verlangt optional mindestens 10 % Medianverkürzung;
ohne diese Option protokolliert der Test auch kleinere Gewinne. API-Aufrufer können
vor erster Verwendung `Reader.ConfigureExpertLookahead(false)` setzen.

Bestanden: vollständiges Paket `./tests`, CLI-Tests und zweifache Race-Prüfung
der Experten-Mapping-/MoE-Tests. `go test ./...` ist unabhängig davon durch die
vorhandene, unversionierte `forward/worker_pool_test.go` blockiert: Sie referenziert
entfernte Workerpool-APIs. Diese Datei wurde weder gelöscht noch deaktiviert;
der alte Pool wurde nicht wieder eingebaut.

### Historisches Experiment: kleine Konstanten und dauerhafte Worker

Die folgenden Ergebnisse beschreiben einen früheren, inzwischen entfernten
Implementierungsstand, nicht die aktuelle Vorlauf-Optimierung.

**Das Ziel von 10 % kürzerer Inferenzzeit wurde auf diesem PC nicht erreicht.**
Die Kombination war langsamer und blieb deshalb damals standardmäßig ausgeschaltet.
Es wird kein Geschwindigkeitsgewinn behauptet.

Damals implementiert waren:

- Ein Cache dekodierter Normgewichte, Attention-Biases/-Sinks und Router-Biases.
  Diese kleinen Vektoren werden bei der Initialisierung validiert und geladen.
  Experten-Biases werden nur für tatsächlich ausgewählte Experten dekodiert.
  FIFO-Verdrängung begrenzt die gespeicherten Float-Daten auf insgesamt 16 MiB;
  Verwaltungsdaten und bestehende Scratch-/KV-Puffer kommen hinzu.
  Große Matrizen bleiben dateibasiert und werden nicht zusätzlich dekodiert gespeichert.
- Ein Engine-eigener Workerpool für Q4_0/Q8_0 und ausgewählte MXFP4-Projektionen.
  Der aufrufende Thread rechnet mit; die Arbeitszahl wird durch `GOMAXPROCS` begrenzt.
  Vor dem Freigeben eines Mappings werden auch bei Abbruch alle Worker abgewartet.
  `Engine.Close()` beendet den Pool, ist idempotent und schließt nicht den fremden Reader.
  Aufrufende Anwendungen müssen ihn vor `Reader.Close()` schließen.

### Messung am echten Modell

Intel Core i7-4790T, 15,5 GiB RAM, vier Compute-Threads, 64-MiB-Fenster,
GPT-OSS-120B mit den beiden vorhandenen Q4_0-GGUF-Shards.
Prompt: `Hallo, sag nur Ja!` (15 Prompt-Tokens), Ausgabelimit fünf Tokens.
Das Modell erzeugt tatsächlich zwei Tokens (`30109`, `13`), Text `Ja.`,
und beendet anschließend mit EOS. Beide Varianten leisten dieselbe Arbeit.

Abschließender Vergleich: je ein Aufwärmlauf, danach drei gepaarte Läufe in
wechselnder Reihenfolge; jeweils neuer Reader, neue Engine und frischer KV-Zustand.
Gemessen wurde der echte `Engine.Generate`-Pfad wie in `main`, einschließlich
Textdekodierung, aber ohne Terminalausgabe. Initialisierung/Prompt-Encoding liegen
außerhalb der Inferenzzeit. `context.Background()` entspricht dem CLI-Aufruf.

| Variante | Median | Bereich der drei Messläufe |
| --- | ---: | ---: |
| Bisheriger Pfad | 29,429 s | 29,023–29,476 s |
| Cache + dauerhafte Worker | 31,872 s | 31,761–31,896 s |

Damit ist die Kombination **8,30 % langsamer**, nicht 10 % schneller.
Tokenfolgen, Texte und Endaktivierungen waren in allen Läufen identisch.
Der Test prüft zusätzlich einen Hash der belegten KV-Positionen; dieser Check
wurde nach der obigen Messreihe ergänzt und im anschließenden Kontrolllauf geprüft.

Ein vorheriger getrennter Versuch deutet auf den Workerpool als Regression hin:
Cache allein lag ungefähr beim Ausgangspfad, Pool allein war langsamer.
Das ist kein belastbarer Nachweis eines Einzelgewinns für den Cache.
Auch gepufferte Aufträge und Mitrechnen des aufrufenden Threads beseitigten die
Regression nicht. Eine genauere Scheduler-/I/O-Ursache ist noch nicht nachgewiesen.

Der OS-Dateicache wurde **nicht geleert**. Pro Lauf wurden ungefähr 20 Millionen
512-Byte-Eingabeblöcke und über 8.000 Major Page Faults gemeldet.
Das ist kein vollständig RAM-warmer und kein kontrollierter Kaltcache-Vergleich.
Die Ergebnisse gelten für diesen PC, dieses Modell und diesen kurzen Prompt;
lange Prompts und andere Hardware wurden nicht umfassend vermessen.

Rohprotokolle der lokalen Experimente liegen unter `bin/optimization-generate-*.log`
(lokale, gegebenenfalls von Git ignorierte Messartefakte).

Die damaligen experimentellen CLI-Schalter und Engine-Optionen existieren nicht
mehr. `TestGenerateModelPerformance` vergleicht jetzt ausschließlich den
Experten-Vorlauf; die alten Rohprotokolle bleiben als historische Evidenz erhalten.