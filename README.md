# Attention_Anchors
> The main purpose of this repo is to implement **Attention applied GRU based Encoder-Decoder Model** in three NLG tasks from scratch and measure its performance.
Each task is Neural Machine Translation, Dialogue Generation, Abstractive Text Summarization. The model architecture has implemented by referring to the famous **Neural Machine Translation by Jointly Learning to Align and Translate** paper, and WMT14, Daily-Dialogue, Daily-CNN datasets have used for each task.
Machine translation and Dialogue generation deal with relatively short sequences, but summarization task covers long sequences. Since it is difficult to properly handle long sentences with only the basic Encoder-Decoder structure, hierarchical encoder structure is used for summary task.
Except for that, all configurations are the same for the three tasks.

<br>
<br>

## Model desc
The main idea of Attention Mechanism came from Human's Brain Cognition Process. People live with a variety of information, but when faced with a specific problem, people usually focus on the information needed to solve the problem. We call this as an **Attention**. The Architecture also use Encoder-Decoder architecture just like **Sequence-to-Sequence** did, but the difference is that the Decoder uses simplified Badanau Attention Operation to make predictions. By using Attention Mechanism, the model could avoid Bottle Neck problem, which results in Better performances in Quantative and Qualitive Evaluation at the same time.

<br>
<br>

## Configurations
| &emsp; **Vocab Config**                            | &emsp; **Model Config**                 | &emsp; **Training Config**               |
| :---                                               | :---                                    | :---                                     |
| **`Vocab Size:`** &hairsp; `30,000`                | **`Input Dimension:`** `30,000`         | **`Epochs:`** `10`                       |
| **`Vocab Type:`** &hairsp; `BPE`                   | **`Output Dimension:`** `30,000`        | **`Batch Size:`** `32`                   |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` &emsp; | **`Embedding Dimension:`** `256` &emsp; | **`Learning Rate:`** `1e-3`              |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]`        | **`Hidden Dimension:`** `512`           | **`iters_to_accumulate:`** `4`           |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]`        | **`N Layers:`** `2`                     | **`Gradient Clip Max Norm:`** `1` &emsp; |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]`        | **`Drop-out Ratio:`** `0.5`             | **`Apply AMP:`** `True`                  |

<br>
<br>


## Results
> **Training Results**

<center>
  <img src="https://user-images.githubusercontent.com/71929682/201269096-2cc00b2f-4e8d-4071-945c-f5a3bfbca985.png" width="90%" height="70%">
</center>


</br>

> **Test Results**

</br>
</br>


## How to Use
**First clone git repo in your local env**
```
git clone https://github.com/moon23k/Attention_Anchors
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, nmt, dialog, sum]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [nmt, dialog, sum] -mode [train, test, inference] -search [greedy, beam]
```


<br>
<br>

## Reference
