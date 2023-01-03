# LongScientificFormer

For encoding a text longer than 512 tokens, for example 800. Set max_pos to 800 during both preprocessing and training.

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)


## Data Preparation 
### Option 1: download the processed data

[Pre-processed data](https://drive.google.com/file/d/1xYHXYoQBa7DJVrq0ePly58ioq2EmmVG8/view)

Put all files into `raw_data` directory. The structure of raw_data folder is 

    root
    ├── logs
    ├── models
    ├── raw_data
    │   ├── 0
    │   ├── 1
    │   ├── 2
    │   ├── 3
    │   ├── 4
    │   ├── ...
    ├── results
    ├── src
    ├── stanford-corenlp-4.2.2
    ├── README.md
    ├── requirements.txt


####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-4.2.2` directory. 
#### step 3. extracting sections from GROBID XML files

```
python preprocess.py -mode extract_pdf_sections -log_file ../logs/extract_section.log
```

#### step 4. extracting text from TIKA XML files

```
python preprocess.py -mode get_text_clean_tika -log_file ../logs/extract_tika_text.log
```

#### step 5. Tokenize texts from papers and slides using stanfordCoreNLP

```
python preprocess.py -mode tokenize  -save_path ../temp -log_file ../logs/tokenize_by_corenlp.log
```


####  Step 6. Extract source, section, and target from tokenized files 

```
python preprocess.py -mode clean_paper_jsons -save_path ../json_data/  -n_cpus 10 -log_file ../logs/build_json.log
```


#### Step 7. Generate BERT `.pt` files from source, sections and targets

```
python preprocess.py -mode format_to_bert -raw_path ../json_data/ -save_path ../bert_data  -lower -n_cpus 40 -log_file ../logs/build_bert_files.log
```


## Model Training

**First run: For the first time, you should use single-GPU, so the code can download the BERT model. Use ``-visible_gpus -1``, after downloading, you could kill the process and rerun the code with multi-GPUs.**

### Train

```
python train.py  -ext_dropout 0.1 -lr 2e-3  -visible_gpus 1,2,3 -report_every 200 -save_checkpoint_steps 1000 -batch_size 1 -train_steps 100000 -accum_count 2  -log_file ../logs/ext_bert -use_interval true -warmup_steps 10000
```
To continue training from a checkpoint
```
python train.py  -ext_dropout 0.1 -lr 2e-3  -train_from ../models/model_step_99000.pt -visible_gpus 1,2,3 -report_every 200 -save_checkpoint_steps 1000 -batch_size 1 -train_steps 100000 -accum_count 2  -log_file ../logs/ext_bert -use_interval true -warmup_steps 10000
```
### Test

```
python train.py -mode test  -test_batch_size 1 -bert_data_path ../bert_data -log_file ../logs/ext_bert_test -test_from ../models/model_step_99000.pt -model_path ../models -sep_optim true -use_interval true -visible_gpus 1,2,3 -alpha 0.95 -result_path ../results/ext 
```

## Deployment

The model is delpoyed in a Streamlit app.

First, at the root folder, install [Grobid](https://grobid.readthedocs.io/en/latest/Install-Grobid/) for pdf extraction. You can run the following command in ther terminal to install Grobid:
```
git clone https://github.com/kermitt2/grobid.git
cd grobid
./gradlew clean install
```

Then, you run the GROBID Service API with the command:
```
./gradlew run
```

Next, to run the Streamlit application, please **open a new terminal** and run the following command from root directory:
```
cd src
streamlit run app.py
```
