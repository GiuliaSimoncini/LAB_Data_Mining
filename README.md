# Cluster-LLM per Real-Time Time-Series Anomaly Detection

Il notebook realizzato replica la pipeline sperimentale dell'articolo **Cluster-LLM: Adaptive Real-Time Time-Series Anomaly Detection Using LLMs** (Zhu et al., CSIS-IAC 2025), combinando *clustering* non supervisionato di serie temporali e inferenza di baseline adattive tramite logica ispirata al *Chain-of-Thought* (CoT) prompting, per il rilevamento di anomalie in tempo reale. Il notebook include inoltre un'appendice che sostituisce la logica deterministica del CoT con chiamate reali a un LLM locale eseguito tramite **Ollama**.

**Autore**: *Giulia Simoncini*

## Setup

Per eseguire il notebook è necessario clonare la repository e poi estrarre i dati da `data.zip`.

La struttura deve essere:

```
cartella_laboratorio/
    data/
    notebook.ipynb
```

### Requisiti

Per eseguire il corpo principale del notebook in esame occorre:
- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `scikit-learn` (K-means, Calinski-Harabasz, LOF)
- `torch` (opzionale, in quanto servirebbe per eseguire LSTM-VAE/DONUT con la vera architettura ricorrente; in assenza di `torch` viene usato un *fallback* numpy/SVD)

Per eseguire anche l'Appendice con chiamate reali al LLM occorre inoltre:
- la libreria `requests` (per interrogare l'API REST di Ollama; se assente, l'appendice ricade automaticamente sulla logica deterministica)
- [Ollama](https://ollama.com) installato e in esecuzione (`ollama serve`), raggiungibile su `http://localhost:11434`
- il modello scaricato con `ollama pull deepseek-r1:8b`

Se questi requisiti non sono soddisfatti, l'appendice resta comunque eseguibile end-to-end, ma ogni chiamata al LLM ricade automaticamente sulla stessa logica statistica deterministica usata nel corpo principale, con un avviso di fallback stampato a schermo.

## Dataset

Il notebook utilizza due dataset pubblici, entrambi presenti nella cartella `data.zip`. È necessario che tali dataset si trovino nella cartella `data/`, descritta in precedenza, per la corretta esecuzione del notebook. I dataset utilizzati sono KPI e Yahoo, e sono impiegati sia nel corpo principale che nell'Appendice.

La tabella seguente riporta, per ciascun dataset, sia le statistiche del paper originale (Table III) sia quelle effettivamente misurate sui dati caricati nel notebook (Sezione EDA):

| Dataset | Fonte | Entità | Punti totali | Anomalie |
|---|---|---|---|---|
| **KPI** (AIOps competition) | Paper (Table III) | 29 | 5.922.913 | 134.114 (2,26%) |
| **KPI** (`./data/kpi/`, AIOps competition) | Notebook (dati reali caricati) | 29 | 3.004.066 | 79.554 (2,65%) |
| **Yahoo** (S5, Yahoo Labs) | Paper (Table III) | 367 | 572.966 | 3.896 (0,68%) |
| **Yahoo** (`./data/yahoo/`, S5, Yahoo Labs) | Notebook (dati reali caricati) | 67 | 94.866 | 1.669 (1,76%) |

Come si può notare, il numero di entità e punti effettivamente disponibili in `data.zip` è inferiore a quello del dataset originale dell'articolo di riferimento (in particolare per Yahoo, 67 entità contro le 367 del paper), tuttavia, le percentuali di anomalie risultano comunque dello stesso ordine di grandezza di quelle riportate nel paper.

Qualora i file reali non siano presenti nella cartella `data/`, il notebook genera automaticamente dati sintetici con struttura statistica equivalente, in modo da poter essere eseguito end-to-end.

## Struttura del Notebook

Il notebook è suddiviso nelle seguenti sezioni:

1. **Setup e import**, che contiene l'import delle librerie utilizzate e il settaggio dei seed per la riproducibilità;
2. **Dataset e caricamento**, che contiene le funzioni di caricamento dei dataset KPI/Yahoo (reali o sintetici);
3. **Exploratory Data Analysis (EDA)**, che riporta le statistiche per entità e quelle aggregate, nonché il confronto con la Table III del paper;
4. **Preprocessing**, che riporta la gestione timestamp, l'interpolazione, lo smoothing e l'arricchimento data/ora;
5. **Clustering**, che riporta le feature giornaliere su 3 finestre orarie, l'algoritmo K-means con la selezione di *k* tramite l'indice di Calinski-Harabasz;
6. **Baseline adattive via CoT**, che riporta i 3 step (z-score, *fitting* Gaussiano ±5 min, fusione) e la *confidence score* per i cluster;
7. **Rilevamento delle anomalie in tempo reale** (ossia l'Algoritmo di Real-Time Anomaly Detection), che riporta lo scoring pesato dalla confidenza dei cluster;
8. **Metriche di valutazione**, ovvero Precision, Recall, F1-score, latenza;
9. **Pipeline end-to-end**, ossia l'esecuzione completa su tutti i dataset e il confronto con la Table IV del paper;
10. **Metodi di confronto**, che analizza i metodi LOF, LSTM-VAE, DONUT, SR-CNN (versioni semplificate);
11. **Tabella e grafici riassuntivi**, ossia la replica della Table IV del paper.

Segue infine un'**Appendice**, che ripropone i macro-passi analoghi a quelli sopra descritti, sostituendo il solo modulo di baseline (Sezione 6) con la sua controparte realmente inferita da un LLM locale (`deepseek-r1:8b` via Ollama):

1. **EDA delle serie di esempio**, ossia le statistiche delle due sole serie (una per KPI, una per Yahoo) usate nel resto dell'appendice, a confronto con la media dell'intero dataset di appartenenza e con i valori presenti nel paper di riferimento;
2. **Implementazione con Ollama**, che comprende il client REST verso Ollama, i tre prompt del CoT (z-score, fit Gaussiano orario, fusione), la pipeline con *fallback* automatico sulla logica deterministica, la visualizzazione delle baseline e del rilevamento sulle serie d'esempio, e l'esecuzione end-to-end (di default limitata alle due serie d'esempio per restare eseguibile in tempi ragionevoli; impostando `RUN_ON_FULL_DATASET = True` la pipeline via LLM viene invece eseguita su tutte le entità, replicando le condizioni della Table IV del paper);
3. **Tabella e grafici riassuntivi**, che aggiungono la riga `Cluster-LLM (Ollama deepseek-r1:8b)` alla tabella comparativa della Sezione 11;
4. **Confronto diretto** tra la variante CoT deterministica (Sezione 6) e quella con CoT realmente inferito dal LLM, a parità di preprocessing, clustering e algoritmo di detection.

## Nota Importante

Il **corpo principale del notebook** (Sezioni 1-11) **non effettua chiamate reali** al modello `deepseek-r1:8b` via Ollama (ossia il LLM usato nel paper originale per l'inferenza delle baseline). Al posto di tali chiamate, i 3 step del CoT sono implementati mediante una logica statistica deterministica equivalente, che riproduce la struttura del metodo ma che non garantisce valori identici alla Table IV del paper. L'andamento qualitativo complessivo (ovvero alta precisione, recall contenuta e bassa latenza) risulta comunque coerente con i risultati riportati dall'articolo di riferimento.

L'**Appendice** effettua invece chiamate reali a `deepseek-r1:8b` tramite Ollama, se disponibile: ogni cluster richiede 3 chiamate al LLM (una per step del CoT). Su un modello da 8B eseguito localmente, replicare l'esecuzione su tutte le 29+67 entità dei due dataset (`RUN_ON_FULL_DATASET = True`) può richiedere da decine di minuti a diverse ore a seconda dell'hardware disponibile; per questo motivo, di default (`RUN_ON_FULL_DATASET = False`), la pipeline via LLM viene eseguita solo sulle due serie d'esempio già usate nel resto dell'appendice. In assenza di un'istanza Ollama attiva (o del modello scaricato), ogni chiamata ricade automaticamente sulla stessa logica deterministica del corpo principale, con un avviso di fallback stampato a schermo, garantendo comunque l'esecuzione end-to-end dell'intera appendice.

## Risultati Principali

La seguente tabella riporta il confronto dei risultati ottenuti dall'approccio Cluster-LLM (variante deterministica, corpo principale del notebook) sia nel caso del notebook realizzato che da parte dell'articolo di riferimento.

| Metodo | KPI F1 | KPI Precision | KPI Time(s) | Yahoo F1 | Yahoo Precision | Yahoo Time(s) |
|---|---|---|---|---|---|---|
| Notebook realizzato | 0,632 | 0,853 | 28,9 | 0,358 | 0,366 | 11,4 |
| Paper di riferimento | 0,642 | 0,985 | 283 | 0,537 | 0,885 | 23 |

Dai risultati ottenuti, si evince che Cluster-LLM risulta il metodo con la migliore precisione su entrambi i dataset e, insieme a SR-CNN, quello con i tempi di esecuzione più contenuti rispetto a LOF, LSTM-VAE e DONUT (tutti oltre i 230s su KPI).

I risultati della variante con CoT realmente inferito dal LLM (Appendice) dipendono dalla disponibilità di un'istanza Ollama locale con il modello `deepseek-r1:8b` e, se eseguita solo sulle due serie d'esempio di default, non sono direttamente confrontabili con la Table IV del paper (calcolata sull'intero dataset); si veda l'Appendice per il confronto diretto con la variante deterministica.

## Bibliografia

B. Zhu, G. Xiong, M. Yuan, Z. Shen, F. Zhu, S. Chen, X. Dong, S. Liu, J. Chen, *Cluster-LLM: Adaptive Real-Time Time-Series Anomaly Detection Using LLMs*, CSIS-IAC 2025.