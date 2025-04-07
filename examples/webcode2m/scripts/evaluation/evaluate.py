import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
sys.path.append(os.path.abspath('.'))
import torch
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
import argparse
from pathlib import Path
from metrics import *
from scripts.evaluation.design2code.visual_score import visual_score_v3,pre_process
import cv2
import pandas as pd
import datetime
from scripts.train.vars import SEED
from PIL import Image
import time
from tools.processor import MultiProcessor    
import multiprocessing
import signal

def html_sim_scores(html1_path, html2_path):   
    with open(html1_path, "r") as f:
         html1 = f.read()
    with open(html2_path, "r") as f:
         html2 = f.read()
    sys.setrecursionlimit(6000)
    bleu, rouge = bleu_rouge(html1, html2)
    tree_bleu, tree_rouge_1 = dom_sim(html1, html2)
 
    return (bleu, rouge, tree_bleu, tree_rouge_1)

def image_sim_scores(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(
        img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )    

    mse_value = mse(img1, img2)
    ssim_value = ssim(img1, img2)
    clip_sim_value = clip_sim(Image.open(img1_path), Image.open(img2_path), 'cpu')
    return mse_value, ssim_value, clip_sim_value

def genertor0(input_dir:Path, output_dir:Path):
    for file in os.listdir(input_dir):
        pred_html_origin=input_dir/f'{file}/prediction.html'
        if not pred_html_origin.exists():
            pred_html_origin=input_dir/f'{file}/pred.html'
        pred_html = output_dir/f'{file}/prediction.html'
        imgs = input_dir/f'{file}/*.png'
        pred_html.parent.mkdir(exist_ok=True, parents=True)
        os.system(f'cp {str(imgs)} {str(pred_html.parent)}/') 
        os.system(f'cp {str(pred_html_origin)} {str(pred_html)}')   
        print(f"cp {str(imgs)} {str(pred_html.parent)}/")
        try:    
            pre_process(str(pred_html))
        except Exception as e:
            print(f"fail to prreprocess: {e}")
            continue            
        answer_html=input_dir/f'{file}/answer.html'
        pred_screenshot=output_dir/f'{file}/prediction.png'
        answer_screenshot=output_dir/f'{file}/answer.png'
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answer_html)} --output {str(answer_screenshot)}")
        os.system(f"python scripts/evaluation/html2screenshot.py --input {str(pred_html)} --output {str(pred_screenshot)}")
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def genertor1(input_dir, output_dir:Path):
    preds_html_dir = input_dir / "preds/html"
    preds_html_dir.mkdir(exist_ok=True, parents=True)
    preds_screenshot_dir = input_dir / "preds/screenshot"
    preds_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_screenshot_dir = input_dir / "answers/screenshot"
    answers_screenshot_dir.mkdir(exist_ok=True, parents=True)
    answers_html_dir = input_dir / "answers/html"
    answers_html_dir.mkdir(exist_ok=True, parents=True)
    print("Taking screenshot of origin htmls ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(answers_html_dir)} --output {str(answers_screenshot_dir)}")
    print("Taking screenshot of predictions ...")
    os.system(f"python scripts/evaluation/html2screenshot.py --input {str(preds_html_dir)} --output {str(preds_screenshot_dir)}")
    for file in tqdm(os.listdir(preds_html_dir)):
        pred_html = preds_html_dir/ file
        pred_screenshot = preds_screenshot_dir/ f"{file.split('.')[0]}.png"
        answer_html = answers_html_dir/ file
        answer_screenshot = answers_screenshot_dir/ f"{file.split('.')[0]}.png"
        yield pred_html,pred_screenshot,answer_html,answer_screenshot

def eval_work(data, out_df):
    pred_html,pred_screenshot,answer_html,answer_screenshot = data
    # if not pred_screenshot.exists() or not answer_screenshot.exists():
    #     print(f"Screenshot file not exits:\n {str(pred_screenshot)}.\n{str(answer_screenshot)}.")
    #     return
    # import pdb;pdb.set_trace()
    bleu, rouge, tree_bleu, tree_rouge_1 = html_sim_scores(answer_html, pred_html)
    mse_value, ssim_value, clip_sim = image_sim_scores(str(answer_screenshot), str(pred_screenshot))
    try:
        _, _, block_match, text_match, position_match, text_color_match, clip_score = \
            visual_score_v3(str(answer_html), str(pred_html), str(answer_screenshot), str(pred_screenshot), str(Path(pred_screenshot).parent), device="cpu")
    except Exception as e:
        print(f"visual_score_v3 error: {e}")
        block_match, text_match, position_match, text_color_match, clip_score = 0, 0, 0, 0, 0
    print(f"{str(answer_html)}, {str(pred_html)},{bleu},{rouge},{tree_bleu}, {tree_rouge_1},{mse_value},{ssim_value},{clip_sim},\
                    {block_match}, {text_match}, {position_match}, {text_color_match}, {clip_score}\n")
    with open(out_df, "a+") as f_csv:
        f_csv.write(f"{str(answer_html)}, {str(pred_html)},{bleu},{rouge},{tree_bleu}, {tree_rouge_1},{mse_value},{ssim_value},{clip_sim},\
                    {block_match}, {text_match}, {position_match}, {text_color_match}, {clip_score}\n")
        #    {block_match},{text_match},{position_match},{text_color_match},{clip_score}\n")

def eval(input_dir, output_dir, generator_choice):
    generator_map={'0':genertor0,'1':genertor1}
    generator=generator_map[generator_choice]
    device = 'cuda'
    torch.manual_seed(SEED)    
    
    # take screenshots, the playwright has to work in the main process.    
    # caculate all the metrics
    input_dir = Path(input_dir)
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d%H%M%S")
    output_dir = Path(output_dir) / f"eval_{input_dir.name}_{time_string}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Caculating metrics:")
    out_df = output_dir / "metrics_result.csv"
    with open(out_df, "w") as f_csv:
        f_csv.write("origin,pred,bleu,rouge,tree_bleu,tree_rouge_1, mse_value,ssim_value, clip_sim, block_match, text_match, position_match, text_color_match, clip_score\n")
    tbar = tqdm(total=len(os.listdir(input_dir))) 
    def cb(res):
        tbar.update(1)   
    pool =  MultiProcessor(12)   
    for data in generator(input_dir,output_dir):
        pool.add_task(eval_work, (data,str(out_df)), cb)
    pool.shutdown()
    df = pd.read_csv(out_df)
    for c in df.columns:
        if c not in ["origin","pred"]:
            print(f"{c}:{df[c].mean():.4f}")

def generator(task):
    for data in task:
        yield data

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Process two path strings.')
    # # Define the arguments
    # parser.add_argument('--input', "-i", type=str)  
    # parser.add_argument('--output', "-o", type=str) 
    # parser.add_argument('--generator', "-g", type=str, choices=['0','1'], default='0') 
    # # Parse the arguments
    # args = parser.parse_args()
    # def signal_handler(signal, frame):
    #     print(f'signal {signal} recieved, exit.')         
    #     for p in multiprocessing.active_children():            
    #         # 获取堆栈信息并写入文件
    #         # 杀死子进程
    #         os.kill(p.pid, signal.SIGKILL)
    #         # 退出主进程
    #     os._exit(1)    
    #         # 设置信号处理程序
    # signal.signal(signal.SIGINT, signal_handler)   
    # print(args)
    # eval(args.input, args.output,args.generator)
    from datasets import load_dataset
    import json
    import numpy as np
    # ds = load_dataset("xcodemind/webcode2m_test")
    # split_names = ["short", "mid", "long"]
    # for split_name in split_names:
    #     with open(f'/home/ubuntu/qwen_2_5_vl_outputs_960_{split_name}.json', 'r', encoding='utf-8') as f:
    #         outputs = json.load(f)
    #     split = ds[split_name]
    #     all_tree_bleu = []
    #     all_tree_rough = []
    #     for idx in range(256):
    #         sample = split[idx]
    #         ref = sample['text']
    #         cand = outputs[idx]['generated_text_raw']
    #         cand = cand.split('<!DOCTYPE html>')
    #         if len(cand) > 1:
    #             cand = cand[1].split('```')[0]
    #         else:
    #             cand = cand[0].split('```')[0]

    #         bleu, rouge, tree_bleu, tree_rouge_1 = html_sim_scores_2(ref, cand)
    #         all_tree_bleu.append(tree_bleu)
    #         all_tree_rough.append(tree_rouge_1)
        
    #     print(f'Split: {split_name}, TreeBleu mean: {np.mean(all_tree_bleu)}, var: {np.var(all_tree_bleu)}')

    # print(html_sim_scores("/home/ubuntu/qwen_html/short/0.html" , "/home/ubuntu/webcode2m/short/0.html"))
    ds = load_dataset("xcodemind/webcode2m_test")
    split_names = ["short", "mid", "long"]
    N = 256
    for split_name in split_names:
        with open(f'/home/ubuntu/qwen_2_5_vl_outputs_960_{split_name}.json', 'r', encoding='utf-8') as f:
            outputs = json.load(f)
        out_df = f'/home/ubuntu/qwen_2_5_vl_outputs_960_{split_name}_eval_single_process.csv'
        with open(out_df, "w") as f_csv:
            f_csv.write("origin,pred,bleu,rouge,tree_bleu,tree_rouge_1, mse_value,ssim_value, clip_sim, block_match, text_match, position_match, text_color_match, clip_score\n")
        split = ds[split_name]
        all_tree_bleu = []
        all_tree_rough = []
        cand_dir = f'/home/ubuntu/qwen_html/{split_name}/'
        ref_dir = f'/home/ubuntu/webcode2m/{split_name}/'
        for idx in range(N):
            sample = split[idx]
            ref =  f'{ref_dir}{idx}.html'
            ref_img = f'{ref_dir}{idx}.png'
            cand = f'{cand_dir}{idx}.html'
            # cand = outputs[idx]['generated_text_raw']
            # cand = cand.split('<!DOCTYPE html>')
            # if len(cand) > 1:
            #     cand = cand[1].split('```')[0]
            # else:
            #     cand = cand[0].split('```')[0]
            cand_img = f'{cand_dir}{idx}.png'
            data = (cand, cand_img, ref, ref_img)
            eval_work(data, out_df)

    # multiprocess
    # for split_name in split_names:
    #     out_df = f'/home/ubuntu/qwen_2_5_vl_outputs_960_{split_name}_eval_single_process.csv'
    #     # Write CSV header *once* at the beginning
    #     with open(out_df, "w") as f_csv:
    #         f_csv.write("origin,pred,bleu,rouge,tree_bleu,tree_rouge_1, mse_value,ssim_value, clip_sim, block_match, text_match, position_match, text_color_match, clip_score\n")

    #     # Build up the tasks we’ll evaluate in parallel
    #     tasks = []
    #     cand_dir = f'/home/ubuntu/qwen_html/{split_name}/'
    #     ref_dir = f'/home/ubuntu/webcode2m/{split_name}/'
    #     # split = ds[split_name]

    #     for idx in range(N):
    #         # Prepare the data for each sample
    #         cand = f'{cand_dir}{idx}.html'
    #         cand_img = f'{cand_dir}{idx}.png'
    #         ref = f'{ref_dir}{idx}.html'
    #         ref_img = f'{ref_dir}{idx}.png'
    #         tasks.append((cand, cand_img, ref, ref_img))


    #     tbar = tqdm(total=len(tasks))
    #     def cb(res):
    #         tbar.update(1)   
    #     pool = MultiProcessor()   
    #     for data in generator(tasks):
    #         pool.add_task(eval_work, (data,str(out_df)), cb)
    #     pool.shutdown()
    #     df = pd.read_csv(out_df)
    #     for c in df.columns:
    #         if c not in ["origin","pred"]:
    #             print(f"{c}:{df[c].mean():.4f}")