# Cluster-LLM: Adaptive Real-Time Time-Series Anomaly Detection Using LLMs

Il notebook realizzato replica la pipeline sperimentale dell'articolo **Cluster-LLM: Adaptive Real-Time Time-Series Anomaly Detection Using LLMs** (Zhu et al., CSIS-IAC 2025), combinando *clustering* non supervisionato di serie temporali e inferenza di baseline adattive tramite logica ispirata al *Chain-of-Thought* (CoT) prompting, per il rilevamento di anomalie in tempo reale.

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

Per eseguire il notebook in esame occorre:
- Python 3.x
- `numpy`, `pandas`, `matplotlib`, `scikit-learn` (K-means, Calinski-Harabasz, LOF)
- `torch` (opzionale, in quanto servirebbe per eseguire LSTM-VAE/DONUT con la vera architettura ricorrente; in assenza di `torch` viene usato un *fallback* numpy/SVD)

## Dataset

Il notebook utilizza due dataset pubblici, entrambi presenti nella cartella `data.zip`. È necessario che tali dataset si trovino nella cartella `data/`, descritta in precedenza, per la corretta esecuzione del notebook. I dataset utilizzati sono KPI e Yahoo.

La tabella seguente riporta, per ciascun dataset, sia le statistiche del paper originale (Table III) sia quelle effettivamente misurate sui dati caricati nel notebook (Sezione EDA):

| Dataset | Fonte | Entità | Punti totali | Anomalie |
|---|---|---|---|---|
| **KPI** (`./data/kpi/`, AIOps competition) | Paper (Table III) | 29 | 5.922.913 | 134.114 (2,26%) |
| **KPI** (`./data/kpi/`, AIOps competition) | Notebook (dati reali caricati) | 29 | 3.004.066 | 79.554 (2,65%) |
| **Yahoo** (`./data/yahoo/`, S5, Yahoo Labs) | Paper (Table III) | 367 | 572.966 | 3.896 (0,68%) |
| **Yahoo** (`./data/yahoo/`, S5, Yahoo Labs) | Notebook (dati reali caricati) | 67 | 94.866 | 1.669 (1,76%) |

Come si può notare, il numero di entità e punti effettivamente disponibili in `data.zip` è inferiore a quello del dataset originale del paper (in particolare per Yahoo, 67 entità contro le 367 del paper): questo è dovuto al sottoinsieme di file realmente incluso in questa repository. Le percentuali di anomalie risultano comunque dello stesso ordine di grandezza di quelle riportate nel paper.

Qualora i file reali non siano presenti nella cartella `data/`, il notebook genera automaticamente dati sintetici con struttura statistica equivalente, in modo da poter essere eseguito end-to-end.

## Struttura del notebook

Il notebook è suddiviso nelle seguenti sezioni:

1. **Setup e import**, che contiene l'import delle librerie utilizzate e il settaggio dei seed per la riproducibilità;
2. **Dataset e caricamento**, che contiene le funzioni di caricamento dei dataset KPI/Yahoo (reali o sintetici);
3. **Exploratory Data Analysis (EDA)**, che riporta le statistiche per entità e quelle aggregate, nonché il confronto con la Table III del paper;
4. **Preprocessing** (che corrisponde alla Sez. III.A), che riporta la gestione timestamp, l'interpolazione, lo smoothing e l'arricchimento data/ora;
5. **Clustering** (che corrisponde alla Sez. III.B), che riporta le feature giornaliere su 3 finestre orarie, l'algoritmo K-means con la selezione di *k* tramite l'indice di Calinski-Harabasz;
6. **Baseline adattive via CoT** (che corrisponde alla Sez. III.C), che riporta i 3 step (z-score, *fitting* Gaussiano ±5 min, fusione) e la *confidence score* per i cluster (Eq. 4);
7. **Rilevamento delle anomalie in tempo reale** (ossia l'Algoritmo 2), che riporta lo scoring pesato dalla confidenza dei cluster;
8. **Metriche di valutazione**, ovvero Precision, Recall, F1-score, latenza (Eq. 5-7);
9. **Pipeline end-to-end**, ossia l'esecuzione completa su tutti i dataset e il confronto con la Table IV del paper;
10. **Metodi di confronto**, che analizza i metodi LOF, LSTM-VAE, DONUT, SR-CNN (versioni semplificate);
11. **Tabella e grafici riassuntivi**, ossia la replica della Table IV del paper.

## Nota importante

Il notebook **non effettua chiamate reali** al modello `deepseek-r1:8b` via Ollama (ossia il LLM usato nel paper originale per l'inferenza delle baseline). Al posto di tali chiamate, i 3 step del CoT sono implementati mediante una logica statistica deterministica equivalente, che riproduce la struttura del metodo ma che non garantisce valori identici alla Table IV del paper. L'andamento qualitativo complessivo (ovvero alta precisione, recall contenuta e bassa latenza) risulta comunque coerente con i risultati riportati dall'articolo di riferimento.

## Risultati principali

La seguente tabella riporta il confronto dei risultati ottenuti dall'approccio ClusterLLM sia nel caso del notebook realizzato che da parte dell'articolo di riferimento.

| Metodo | KPI F1 | KPI Precision | KPI Time(s) | Yahoo F1 | Yahoo Precision | Yahoo Time(s) |
|---|---|---|---|---|---|---|
| Notebook realizzato | 0,632 | 0,854 | 35,1 | 0,358 | 0,366 | 17,5 |
| Paper di riferimento | 0,642 | 0,985 | 283 | 0,537 | 0,885 | 23 |

Dai risultati ottenuti, si evince che Cluster-LLM risulta il metodo con la migliore precisione su entrambi i dataset e, insieme a SR-CNN, quello con i tempi di esecuzione più contenuti rispetto a LOF, LSTM-VAE e DONUT (tutti oltre i 290s su KPI).

## Bibliografia

B. Zhu, G. Xiong, M. Yuan, Z. Shen, F. Zhu, S. Chen, X. Dong, S. Liu, J. Chen, *Cluster-LLM: Adaptive Real-Time Time-Series Anomaly Detection Using LLMs*, CSIS-IAC 2025.