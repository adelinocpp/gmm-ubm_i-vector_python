# Rotina para treinamento,teste e validação para comparação de locutores.

Utiliza as técnicas GMM-UBM, i-vector e ~~fuzzy i-vector~~.

__autor:__ Adelino Pinheiro Silva
__email:__ adelinocpp@yahoo.com

Inspirado no trabalho de [Dudans](dudans@kaist.ac.kr), o [SpeakerRecognition_tutorial
](https://github.com/jymsuper/SpeakerRecognition_tutorial)

### Indicando os bancos de dados:

1 - Abrir o arquivo "configure.py" e indicar os caminhos dos arquivos de __treinamento__ na variável "TRAIN_WAV_DIR" e de __testes__  na variável "TEST_WAV_DIR". 

Em cada diretório (TRAIN_WAV_DIR e TEST_WAV_DIR) os subdiretorios serão interpretados como __chaves__ de identificação de locutores.  Dentro de cada subdiretório devem ser armazenados os arquivos "*.wav" (_utterances_) de um mesmo locutor. Exemplo do diretório de treinamento

```
/training_dir
    /0001
        /utterance_01.wav
        /utterance_02.wav         
        /utterance_03.wav  
        ...
    /1005
        /utterance_01.wav
        /utterance_02.wav         
        /utterance_03.wav  
```

Sugiro utilizar o nome dos subdiretorios dos locutores como numeros únicos para cada locutor. Fique atento, se um mesmo locutor estiver no dietório de treinamento e de teste eles precisam ter a mesma __chave__ (nome/número de diretório) ou serão considerados como locutores diferentes (o que pode ser um problema na modelagem LDA). 

O nome do arquivo dentro do diretório não tem tanta importância. Utilize o arquivo de áudio com extensão wav, codificação PCI em 8 kHz e 16 bits.

### Calculando características

Em construção...