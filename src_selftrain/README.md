# Source code for self-training

### Install
```
pip install transformers
pip install datasets
pip install evaluate
```


### Data
We share the pre-processed STAC data [1] in `data/`, where:
- `stac_{subset}_spk.csv` contains complete {subset} relation pairs.
- `combined/al_train700_seed27.csv` contains the seeded 700 pairs that we use for initial finetuning.

*[1] Nicholas Asher, Julie Hunter, Mathieu Morey, Bena- mara Farah, and Stergos Afantenos. 2016. Discourse structure and dialogue acts in multiparty dialogue: the STAC corpus. In Proceedings of the Tenth In- ternational Conference on Language Resources and Evaluation (LREC’16), pages 2721–2727, Portorož, Slovenia. European Language Resources Association (ELRA).*

### Code
- `finetune.py`: we start with a subset of training examples, ranging from [700, 1500, 2500, 5000, 7500]. The checkpoints obtained in this step is used as prediction model in the next step for self-training.
- `self-train`: main self training code, need finetuned BERT checkpoints as input
- You can tweak the code for multiple loop of self-training.


Please consider citing our paper: 
```
@inproceedings{li-etal-2024-discourse,
    title = "Discourse Relation Prediction and Discourse Parsing in Dialogues with Minimal Supervision",
    author = "Li, Chuyuan  and
      Braud, Chlo{\'e}  and
      Amblard, Maxime  and
      Carenini, Giuseppe",
    editor = "Strube, Michael  and
      Braud, Chloe  and
      Hardmeier, Christian  and
      Li, Junyi Jessy  and
      Loaiciga, Sharid  and
      Zeldes, Amir  and
      Li, Chuyuan",
    booktitle = "Proceedings of the 5th Workshop on Computational Approaches to Discourse (CODI 2024)",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.codi-1.15",
    pages = "161--176",
```