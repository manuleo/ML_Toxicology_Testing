## Question for Marco
- Quali databases dovremmo usare?
  - tests, results, species (per raggruppare?), chemicals? (chemicals_carrier? -> per considerare anche il solvente/info su sostanze chimiche)
  - Dosi? 4 database...
- Non capiamo il suffisso *_op* su diverse features
- Link specie <-> organismi in termini di features
- Come raggruppare organismi, come individuare quali saranno le nuove specie da testare? 

## Info
**Features**:
- CAS Number: è in test, possiamo linkarlo a chemicals (solo il nome) o chemicals_carriers (ci dovrebbe dare info sulle sostanze chimiche ch si potrebbero usare come features
  - Dosi: ci potrebbero servire dopo, magari calcolare nuove features (categoriche? Dipende se vengono utilizzate dosi fisse) da lì.
- Organism: si trova in test, abbiamo diverse features disponibili riguardo età, habitat, species_id, peso, lifestage, gender, "characteristics" (just string, many NR). Vanno linkati alle specie per capire secondo cosa potremmo predire in futuro
- Exposure: 
