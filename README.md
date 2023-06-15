# TODD - Thermal Object Detection for autonomous Driving
Questo repositorio contiene il codice per sperimentare i tempi di risposta di una Nvidia Jetson Nano 2GB nel rilevamento oggetti. Nello specifico è stato usato un modello addestrato a riconoscere oggetti da immagini termiche in un elaboratore separato con una GPU più potente e, una volta ottenuto il modello, importato, ottimizzato con NVIDIA TensorRT ed eseguito sulla Jetson Nano.

Vi sono diverse fasi e quindi, anche diverse librerie e framework. Parte dello studio è stato anche quello di individuare il flusso di lavoro più agevole allo stato attuale. In particolare, abbiamo intrapreso due starde di sviluppo: 
- Tensorflow, in questo caso si rimane nello stesso ambiente in tutte le fasi:
    - Importazione della rete preallenata SSD Mobilenet V2 in Tensorflow Hub
    - Addestramento del modello in Tensorflow
    - Esportazione in formato SavedModel 
    - Importazione nella Jeston sempre con Tensorflow 
    - Ottimizzazione con TF-TensorRT
    - Inferenza in Tensorflow
- Pytorch, qui, invece, bisogna passare per il formato ONNX
    - Importazione della rete preallenata SSDLite Mobilenet V3 in Pytorch Hub
    - Addestramento del modello con Pytorch
    - Esportazione in formato ONNX
    - Ottimizzazione in TensorRT
    - Inferenza in TensorRT

## Risultati
I risultati sono stati soddisfacenti per il flusso in Tensorflow. I tempi risposta della Jetson Nano si sono attestati intorno ai 10 FPS. Considerando che è stato testato su una scheda da 60€ e che, solo tre anni fa un esperimento simile è stato intrapreso con una RaspberryPi3 ([Real-Time Human Detection as an Edge Service Enabled by a Lightweight CNN](https://ieeexplore.ieee.org/document/8473387) ). In quel caso il risultato migliore è stato di 1.82 FPS. Questo risultato potrebbe essere ulteriormente migliorato sfruttando l'API in C++ di Tensorflow o, ancora, riuscendo a ottimizzare tutto il modello in TensorRT.

Per quanto riguarda la precisione è stata usat la metrica COCO. Si riporta qui la tabella con i risultati tra le due architetture a confronto
|Metric|IoU|Area|maxDets|SSDLite Mobilenet V3|SSD Mobilenet V2|
|------|---|----|-------|-----|------|
|Average Precision  (AP)|0.50:0.95|all|100|0.076|0.111|
|Average Precision  (AP)|0.50|all|100|0.173|0.3|
|Average Precision  (AP)|0.75|all|100|0.061|0.07|
|Average Precision  (AP)|0.50:0.95|small|100|0.007|0.045|
|Average Precision  (AP)|0.50:0.95|medium|100|0.095|0.2497|
|Average Precision  (AP)|0.50:0.95|large|100|0.486|0.4244|
|Average Recall     (AR)|0.50:0.95 |all|1|0.054|0.089|
|Average Recall     (AR)|0.50:0.95 |all|10|0.147|0.22|
|Average Recall     (AR)|0.50:0.95 |all|100|0.205|0.23|
|Average Recall     (AR)|0.50:0.95 |small|100|0.099|0.126|
|Average Recall     (AR)|0.50:0.95 |medium|100|0.257|0.4452|
|Average Recall     (AR)|0.50:0.95 |large|100|0.640|0.6393|

Di seguito viene mostrato un test effettuato sulla Jetson Nano. Il frame rate è ridotto a circa 3 FPS perchè l'elaborazione del risultato ed il disegno delle bounding box viene effettutato tutto sulla scheda, e questo richiede circa 300 ms.

![video sample](doc/sample-output.gif)

Purtroppo, al momento di redigere questo articolo, non si è stati riusciti a completare lo sviluppo attraverso Pytorch. L'ostacolo è stato dovuto alla conversione attraverso il formato ONNX, infatti, TensorRT ancora non supporta tutti i moduli della rete originale che vengono codificati in tale formato. Per proseguire si sarebbe dovuto intervenire manualmente nel file ONNX prima di fornirlo a TensorRT ma il procedimento sarebbe risultato lungo e complesso. Non è avvenuto lo stesso in Tensorflow, perché TF-TRT si occupa di convertire solo i moduli compatibli lasciando gli altri invariati. Dato il continuo contributo da parte di NVIDIA ed i continui aggiornamenti, ci si aspetta in futuro una conversione sempre più agevole.
