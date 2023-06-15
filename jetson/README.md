# Object detection of thermal images on Jetson Nano
Inferenza su di un modello di object detection addestrato sul dataset di immagini termiche FLIR. 
Due modelli vengono messi a confronto entrambi ottenuti con due framework diferenti: un modello è stato ottenuto in formato saved_model di Tensorflow 2. Si tratta di una SSD Mobilenet V2. Per poter rendere l'inferenza il più efficiente possibile si utilizza TF-TRT. L'altra architettura utilizzata è una SSDLite Mobilenet V3 Large, sfruttando la libreria Pytorch. In questo caso si usa direttamente TensorRt per l'ottimizzazione.

## Componenti
TensorRT è un kit di sviluppo software per ottimizzare l'inferenza di reti neurali artificiali in GPU della famiglia di schede Nvidia. E' presente nel pacchetto software fornito nell'immagine del sistema operativo fornito dal produttore chiamato JetPack. Però, servono altri componenti prima di eseguire l'inferenza.


## Requisiti
Per eseguire il codice è necessario installare le seguenti applicazioni che non sono presenti sulla Jetson:
1. Tensorflow [Installing Tensorflow for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html)
2. Jetson-inference [Deploying Deep Learning](https://github.com/dusty-nv/jetson-inference), contiene codice utile all'inferenza come la gestione dei flussi di immagini da telecamera o da file. Durante l'installazione viene richiesta l'installazione anche di Pytorch.
3. FLIR thermal dataset [FLIR thermal image dataset](https://www.kaggle.com/deepnewbie/flir-thermal-images-dataset)

## Come avviare
Scaricare il dataset di test che si trova nel seguente collegamento: [FLIR thermal image dataset](https://www.kaggle.com/deepnewbie/flir-thermal-images-dataset). Il programma si aspetta di trovare le immagini in images/video.

Se il modello ottimizzato non è già pronto, bisogna convertire quello originale che si trova in my_model/saved_model. 

        python3 convert.py

A questo punto si può eseguire l'inferenza che genererà un file video in uscita

        python3 detect_tf.py


## Risultati
Sono stati misurati i tempi di computazione i tre diversi momenti:
1. pre-processamento, quando viene caricato un frame in memoria e il ridimensionamento del dato in ingresso
2. inferenza, il tempo che impiega Tensorflow ad interrogare il modello
3. post-processamento, essenzialmente il ciclo che itera i risultati e disegna le bounding-box

I risultati mostrano un dispendio irrisorio per il pre-processamento e più consistente per la visualizzazione dei risltati, che sono di circa 200 millisecondi. Mentre l'inferenza inpiega intorno ai 100 millisecondi. Questo corrisponde ad un totale di 3 FPS.

Una considerazione va fatta per il ciclo di disegno delle bbox: per ottenere l'immagine finale si sfrutta la libreria jetson.utils che a sua volta racchiude codice in CUDA, quindi è il massimo dell'efficienza. La stessa cosa non si può dire per l'estrazione dei dati, che si ritiene essere il principale collo di bottiglia. Infati, si tratta di un ciclo for in Python con una lettura di elementi di array all'interno di un dizionario.

Il caricamento del modello TF è molto lento sulla Jetson. Oltre i dieci minuti. Questo è un problema noto come si può trovare nelle discussioni in rete. Qualcuno ha risolto il problema compilando ed installando una versione del pacchetto protobuf compilato in C++ [TensorFlow/TensorRT (TF-TRT) Revisited](https://jkjung-avt.github.io/tf-trt-revisited/). Inoltre, la prima inferenza sul modello è molto lunga, rsultando in una manciata di secondi di frame neri all'inizio del video in uscita.

In ulltima analisi, si vuole confrontare le prestazioni che si ottiene da una generica rete SSD Mobilenet V2 (non allenato sui dati termici) implementata con la libreria jetson-inference. Il risultato è di circa 20 FPS. E' preferibile, almeno alle versioni attuali del pacchetto di applicaione, costruire il modello in Pytorch, perchè esportare in ONNX crea meno problemi di Tensorflow e dall'ONNX si può ottenere un modello ottimizzato in TensorRT pure più performante di TF-TRT. 

![out](out/output_sample.png)

