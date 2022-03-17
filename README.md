# augustine-will

Serveren til semantisk graphtegner baseret på Augustine-Will teksterne

## Sådan virker værktøjet:

Værktøjet har to separate funktioner:

1. Semantisk graph
2. Ordanalyse

### 1. Semantisk graph

- Graphen tegnes baseret på en **word2vec** model fil der skal indsættes til folderen `/dat`.
- Når en bruger taster nogle ord ind til værktøjet de bliver lemmatiseret med lemmatizeren i `latin.py` og anvendt som **seeds** til en semantisk kernel
- Oprettelsen af en semantisk kernel:
  1. _k_ ord der er tættest på **seederne** samles til en liste, de bliver kaldt for kernellens **types**
  2. _m_ ord der er tættest på **types** samles
  3. Alle de her ord sættes sammen til en liste, **types** og **seeds** med store bogstaver
  4. En graph tegnes til alle de ord man får ud af denne process

### 2. Ordanalyse

- `/dat/token_table.csv` anvendes til at se på ordbrug i corpus. Denne fil skal forudberegnes
- **seederne** bliver anvendt som de kommer, de bliver ikke lemmatiseret
- To grapher tegnes:
  1. Ordforbrug i alt af **seederne**
  2. Ordborbrug over tiden enten med absolute tal eller procentvis
