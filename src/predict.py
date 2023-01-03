import os
import json
import torch
import numpy as np
import bisect
import subprocess
import prepro.data_builder as data_builder
from others.utils import clean
import argparse
from models.model_builder import ExtSummarizer

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-load_from", default='../alakimodels/last.pt', type=str)
parser.add_argument("-select_mode", default='greedy', type=str)
parser.add_argument("-shard_size", default=50, type=int)
parser.add_argument('-min_src_nsents', default=20, type=int)
parser.add_argument('-max_src_nsents', default=500, type=int)
parser.add_argument('-min_src_ntokens_per_sent', default=5, type=int)
parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
parser.add_argument('-min_tgt_ntokens', default=50, type=int)
parser.add_argument('-max_tgt_ntokens', default=5000, type=int)
parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('-log_file', default='../logs/slide_gen.log')
parser.add_argument("-large", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-temp_dir", default='../temp')
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)
parser.add_argument("-max_pos", default=10240, type=int) #fix
parser.add_argument("-chunk_size", default=512, type=int) # fix
parser.add_argument("-ext_dropout", default=0.2, type=float)
parser.add_argument("-ext_layers", default=2, type=int, help="number of extractive encoder layers")
parser.add_argument("-ext_hidden_size", default=768, type=int)
parser.add_argument("-ext_heads", default=4, type=int, help="number of attention head in each encoder layer")
parser.add_argument("-ext_ff_size", default=2048, type=int)
# global attention params
parser.add_argument('-global_attention', default=1, type=int, choices=[0,1,2], help=" global attention types:0,1,2. 0: no global attention, 1: global attention at random indices, 2: global attention at the beginning and end of the sections  ")
parser.add_argument('-global_attention_ratio', default=0.2, type=float, help="ratio of global attention indices chosen at random")
args = parser.parse_args()


## load checkpoint
checkpoint = torch.load(args.load_from, map_location=lambda storage, loc: storage)
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']
opt = vars(checkpoint['opt'])
for k in opt.keys():
    if k in model_flags:
        setattr(args, k, opt[k])

## load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# print("device:", device)

model = ExtSummarizer(args, device, checkpoint)
model.eval()

def preprocess(pdf_path):
    # Extract the sections from the PDF
    outfile_path = os.path.join(args.temp_dir, "paper.sections.txt")
    outfile = open(outfile_path, "w")
    for line in data_builder.read_pdf_sections(pdf_path):
        outfile.write(line.strip() + '\n')
    outfile.close()

    with open("mapping_for_corenlp.txt", 'w') as f:
        f.write(outfile_path + '\n')

    # Run CoreNLP
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                'json', '-outputDirectory', args.temp_dir]

    subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    assert os.path.exists(outfile_path + ".json"), "CoreNLP failed to produce output file"

    # make data
    pdf_json = os.path.join(args.temp_dir, "paper.sections.txt.json")
    source = []
    sections = []
    section = 1
    for sent in json.load(open(pdf_json))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        after_tokens = [t['after'] for t in sent['tokens']]
        if args.lower:
            tokens = [t.lower() for t in tokens]
        source.append(tokens)
        sections.append(section)
        if len(after_tokens) > 0 and after_tokens[-1] == '\n':
            section += 1

    source = [clean(' '.join(sent)).split() for sent in source]

    data_json = {'src': source, 'sections': sections}

    # format_to_bert
    corpus_type = "test"
    json_f = os.path.join(args.temp_dir, "data.json")

    def format_to_bert(args, corpus_type, data):
        is_test = corpus_type == 'test'
        bert = data_builder.BertData(args)
        source, sections = data['src'], data['sections']
        # greedily selects the top 3 sentences and labels them as 1
        summary_size = int(0.2 * len(source))
        if args.lower:
            source = [' '.join(s).lower().split() for s in source]
        b_data = bert.preprocess(source, sections, [], [],
                                    use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
                                    is_test=is_test)
        src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt, sections, token_sections = b_data
        b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
                        "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
                        'src_txt': src_txt, "tgt_txt": tgt_txt, "sections": sections, "token_sections": token_sections}

        return b_data_dict

    # clean temp
    os.remove(outfile_path)
    os.remove(outfile_path + ".json")
    os.remove("mapping_for_corenlp.txt")

    data = format_to_bert(args, corpus_type, data_json)
    return data

def post_process(data, device):
    src = data['src']
    sections = data['sections']
    token_sections = data['token_sections']
    segs = data['segs']
    clss = data['clss']

    src_txt = data['src_txt']

    end_id = [src[-1]]
    lastIsCls = False
    if len(src) > args.max_pos-1 and src[args.max_pos-1] == 101:
        lastIsCls = True

    src = src[:-1][:args.max_pos - 1] + end_id
    segs = segs[:args.max_pos]

    token_sections = token_sections[:args.max_pos]
    max_sent_id = bisect.bisect_left(clss, args.max_pos)
    clss = clss[:max_sent_id]
    sections = sections[:max_sent_id]
    if lastIsCls:
        clss = clss[:max_sent_id-1]
        sections = sections[: max_sent_id-1]

    def _pad(data, pad_id, width=-1):
        if width == -1:
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    src = torch.tensor(_pad([src], 0)).to(int)
    segs = torch.tensor(_pad([segs], 0)).to(int)
    token_sections = torch.tensor(_pad([token_sections], 0)).to(int)
    clss = torch.tensor(_pad([clss], -1)).to(int)
    sections = torch.tensor(_pad([sections], 0)).to(int)
    mask_src = ~ (src == 0).to(int)
    mask_cls = ~ (clss == -1)
    clss[clss == -1] = 0

    # move to device
    src = src.to(device)
    segs = segs.to(device)
    token_sections = token_sections.to(device)
    mask_src = mask_src.to(device)
    clss = clss.to(device)
    sections = sections.to(device)
    mask_cls = mask_cls.to(device)

    return {'src': src, 'segs': segs, 'clss': clss, 'sections': sections, 'token_sections': token_sections, 'mask_src': mask_src, 'mask_cls': mask_cls}, src_txt

def predict(args, model, pdf_path, device):

    extracted_data = preprocess(pdf_path)
    data, src_txt = post_process(extracted_data, device)
    mask_cls = data['mask_cls']
    sent_scores, mask = model(**data)
    batch_size, sent_count = mask_cls.shape
    sent_scores = sent_scores[:, :sent_count]  # remove padded items from returned scores
    sent_scores = sent_scores.cpu().data.numpy()
    selected_ids = np.argsort(-sent_scores, 1)[0]
    # sort src text by section
    src_txt = [src_txt[i] for i in selected_ids]

    return selected_ids, src_txt

pdf_path = "../examples_data/Paper_cf.tei.xml"
selected_ids, src_txt = predict(args, model, pdf_path, device)

# for txt in src_txt:
#     print(txt[:50])
