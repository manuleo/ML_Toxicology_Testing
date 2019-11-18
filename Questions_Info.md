## Question for Marco
- Quali databases dovremmo usare?
  - tests, results, species (per raggruppare?), chemicals? (chemicals_carrier? -> per considerare anche il solvente/info su sostanze chimiche)
  - Dosi? 4 database possibili, conc_type all'interno. Concentrazioni per 3 diverse dosi? Vanno linkate ai results?
- Non capiamo il suffisso *_op* (Operation) su diverse features. Operation ha tanti valori nulli. Ci serve?
- Link specie <-> organismi in termini di features (Come definire una specie basata su quali features degli organismi)
- Features degli organismi? età, habitat, species_id, peso, lifestage, gender. Ce ne sono altre? Analizzando le features degli organismi ci siamo accorti che tanti riportano NR in molte caratteristiche. Come ci lavoriamo? (Da considerare anche che vanno riviste come speci)
- I missing values **NC** sono un sacco nel test tra organismi e altre variabili (e ci sono tante corrispondenze NR - NC a incrociare che aumentano il numero dei NaN totali
- **Dato che organism da tanti problemi** -> fare un join con species_number e considerare solo le caratteristiche delle speci? (Features possibili: 'kingdom', 'class', 'tax_order', 'family', 'genus', 'species', 'subspecies', 'variety', **ecotox_group**). 
- Come raggruppare organismi, come individuare **quali saranno le nuove specie da testare**?
- **Quali saranno i nuovi input? Bisogna classificare data una specie/ecotox_group e le caratteristiche del chemicals? Dove dovremmo prenderle?**
- Come comportarci con le unità di misura dell'exposure length. Come le raggruppiamo? (E.g. per i top: hv=harves, gs=growing season)
  Inoltre, **metà sono non definite. Stessa cosa per le corrispondenti misurazioni** -> Che facciamo con questo metà dataset
- Che vuol dire major negli obiettivi? 
- Che cosa intendiamo per concentration? Ci sono migliaia di diverse unità di misura, come dovremmo confrontarle?
- Come creare in maniera intelligente un db unico per fare le prove, su cosa possiamo basarci.


## Info
**Features**:
- CAS Number: è in test, possiamo linkarlo a chemicals (solo il nome) o chemicals_carriers (ci dovrebbe dare info sulle sostanze chimiche ch si potrebbero usare come features
  - Dosi: ci potrebbero servire dopo, magari calcolare nuove features (categoriche? Dipende se vengono utilizzate dosi fisse) da lì.
- Organism: si trova in test, abbiamo diverse features disponibili riguardo età, habitat, species_id, peso, lifestage, gender, "characteristics" (just string, many NR). Vanno linkati alle specie per capire secondo cosa potremmo predire in futuro
- Exposure length: in test. Exposure_duration è un numero ma le unità di misura sono molto diverse tra loro, includono un grande insieme di valori possibili (89) e molti sono legati a caratteristiche che sono difficili da riportare. **Problema: metà dataset non definite**
- Effect type: in results (endpoint) -> endpoint_code per descrizione (E' il nostro label). 298 possibili classi da predirre (NR -> Non Riportato, potremmo eliminarlo) 
- Concentration type: in result? Diverse features conc1, conc2, conc3 mean, max, type, unit. Cosa dovremmo ricavare da queste features?                       **Solo conc1 ha una buona percentuale di valori utilizzabili**
                      in dosi? Concentrazioni per 3 diverse dosi?
