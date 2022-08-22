import json
import requests
import csv
import argparse
import os
import time
import torch
from datetime import datetime
import re
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from word2number import w2n
device = 0 if torch.cuda.is_available() else -1

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

def open_file(file_name):
  file = open(file_name, "r")
  data = json.load(file)
  file.close()

  return data


def replace_hash(old, decomp_list):
    print(old)
    list_of_hash = []
    for c in range(len(old)):
        if old[c] == "#":
            list_of_hash.append((c,old[c+1]))
                
    list_of_hash.reverse()
    for k in list_of_hash:
        index, num = k[0], k[1]
        try:
            old = old.replace(old[index:index+2], decomp_list[int(num)-1])
        except Exception:
            old = old.replace(old[index:index+2], str(decomp_list[int(num)-1]))
    return old
    
def calculate(func):
    
    args = func.split(' ! ')
    ans = 0
    if args[0] == "summation" or args[0] == "addition":
        ans = sum_answer(args[1:])
    if args[0] == "difference" or args[0] == "difference":
        ans = diff_answer(args[1:])
    if args[0] == "greater":
        ans = greater_answer(args[1:])
    if args[0] == "date":
        ans = str(0)
    if args[0] == "lesser":
        ans = lesser_answer(args[1:])
    if args[0] == "division":
        ans = division_answer(args[1:])
    if args[0] == "multiplication":
        ans = multiplication_answer(args[1:])    
    
    return ans

def division_answer(args):
    ans = 0
    for i in range(len(args[:2])):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
    try:
        if len(args) > 2:
            ans = int(args[0]/args[1])
        else:
            ans = args[0]/args[1]
    except Exception:
        ans = 0
    
    return str(ans) 

def multiplication_answer(args):
    ans = 0
    for i in range(len(args[:2])):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
    try:
        ans = args[0]*args[1]
    except Exception:
        ans = 0
    
    return str(ans) 

def greater_answer(args):
    ans = 0
    for i in range(len(args[:2])):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
    
    if len(args) > 2:
        if args[0] > args[1]:
            return str(args[2])
        else:
            return str(args[3])
    else:
        return str(max(args))
    return str(ans) 

def lesser_answer(args):
    ans = 0
    for i in range(len(args[:2])):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
    
    if len(args) > 2:
        if args[0] < args[1]:
            return str(args[2])
        else:
            return str(args[3])
    else:
        return str(min(args))
    return str(ans) 
    
def sum_answer(args):
    ans = 0
    for i in range(len(args)):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
        ans += args[i]
            
    return str(ans)  
        
def diff_answer(args):
    
    for i in range(len(args)):
        try:
            args[i] = w2n.word_to_num(args[i])
        except Exception:
            num = ""
            num = num.join([c for c in args[i] if c.isdigit() or c is '.'])
            try:
                args[i] = float(num)
            except Exception:
                args[i] = 0
        
    args.sort(reverse=True)
    ans = args[0]
    for i in args[1:]:
        ans -= i
            
    return str(ans)


    
def get_answers_roberta(data, dataset_name):
    responses = []
    
    model_name = "deepset/roberta-base-squad2"
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name, device= device)
    p = 0
    ori = []
    for data_point in data:
        p += 1
        print(p)
        final_answers = {}
        
        QA_input = {
                    'question': data_point['question'],
                    'context': data_point['context']
                    }
        original_ans = qa_model(QA_input)
        
        try:
            original_ans = str(w2n.word_to_num(original_ans['answer']))
        except Exception:
            original_ans = str(original_ans["answer"])
        
        if original_ans.lower().strip() in data_point['answer'].lower().strip() or data_point['answer'].lower().strip() in original_ans.lower().strip():
            ori.append(str(data_point["answer"]).lower())
            responses.append({"original": original_ans, "decomposition": original_ans})
            continue
        
         #ans['answer']
        decomp_answers = []
        d_len = len(data_point["decompositions"]) - 1
        for decomposition in data_point["decompositions"]:
            if "#" in decomposition["question"]:
                decomposition["question"] = replace_hash(decomposition["question"], decomp_answers)
            
            
            
            if "!" in decomposition["question"]:
                ans = calculate(decomposition["question"])
                decomp_answers.append(ans)
            else:
                QA_input =  {
                            'question': decomposition['question'],
                            'context': data_point['context']
                            }
                ans = qa_model(QA_input)
                decomp_answers.append(ans['answer'].lower())
                decomposition["generated_answer"] = ans['answer']
                print(decomposition["question"])
            
            if d_len <=0:
                
                if str(data_point["answer"]).lower() in str(original_ans).lower():
                    original_ans = data_point["answer"]
                
                final_answers["original"] = original_ans
                
                if data_point["answer"].lower() in str(decomp_answers[-1]).lower():
                    decomp_answers[-1] = data_point["answer"]
                
                final_answers["decomposition"] = decomp_answers
                ori.append(str(data_point["answer"]).lower())

                responses.append(final_answers)
            
            
                #decomposition["generated_answer"] = ans["answer"]
                data_point["generated_answer"] = original_ans
                
            d_len -= 1
            
    ori_pred = [str(i["original"]).lower() for i in responses]
    
    decomp_pred = []
    
    for i in responses:
        if str(type(i["decomposition"])) == "<class 'list'>":
            decomp_pred.append(i["decomposition"][-1])
        else:
            decomp_pred.append(i["decomposition"])
    #decomp_pred = [str(i["decomposition"][-1]).lower() for i in responses]
    
    response_original = {"true": ori, "prediction": ori_pred}
    response_decomp = {"true": ori, "prediction": decomp_pred}
    
    file = open(dataset_name  +"_roberta_original.json", "w")
    json.dump(response_original, file, indent=4)
    file.close()
    
    file = open(dataset_name +"_roberta_decomposition.json", "w")
    json.dump(response_decomp, file, indent=4)
    file.close()
        
    file = open(dataset_name +"_roberta_trial.json", "w")
    json.dump(data, file, indent=4)
    file.close()
    
    return responses

def compare_answers(file, responses,output_file):
  data = open_file(file)
  for i in range(len(data)):
    original = responses[i]["original"]
    data[i]["generated_answer"] = original
    
    for j in range(len(data[i]["decompositions"])):
        answer = responses[i]["decomposition"][j]
        if type(data[i]["decompositions"][0]) is list:
            data[i]["decompositions"] = data[i]["decompositions"][0]
        data[i]["decompositions"][j]["generated_answer"] = answer
    
    
  file = open(output_file, "w")
  json.dump(data,file,indent=4)
  file.close()
  
def generate_answers(dataset_name, model_name, API_TOKEN):
    file = dataset_name + "/" + dataset_name + ".json"
    
    data = open_file(file)
    if model_name == "roberta":
        responses = get_answers_roberta(data, dataset_name)
        
        #compare_answers(file, responses, dataset_name + "/" + dataset_name + "_roberta_trial.json")

def gold_and_predictions(dataset_name, model_name):
    
    file_name = dataset_name + "/" + dataset_name + "_" + model_name + "_trial.json"
    data = open_file(file_name)
    
    true_answers = []
    generated_answers_original = []
    generated_answers_decomposition = []
    for i in data:
        true_answer = str(i["answer"])
        
        true_answers.append(true_answer)
        generated_answer = str(i["generated_answer"])
        
        if true_answer.lower() in generated_answer.lower():
            generated_answer = true_answer
        generated_answers_original.append(generated_answer)
        
        if type(i["decompositions"][0]) is list:
            i["decompositions"] = i["decompositions"][0]
            print(i["decompositions"][-1])
        
        decomp_answer = str(i["decompositions"][-1]["generated_answer"])
        
        
        
        if true_answer.lower() in decomp_answer.lower():
            decomp_answer = true_answer
        generated_answers_decomposition.append(decomp_answer)
    
    answers = {"true": true_answers, "prediction": generated_answers_original}
    file = open(dataset_name + "/" + dataset_name + "_" + model_name + "_original.json", "w" )
    json.dump(answers, file, indent=4)
    file.close()
    
    answers = {"true": true_answers, "prediction": generated_answers_decomposition}
    file = open(dataset_name + "/" + dataset_name + "_" + model_name + "_decomposition.json", "w" )
    json.dump(answers, file, indent=4)
    file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of Question Decomposition')
    parser.add_argument('--model_name', help='model to be used')
    parser.add_argument('--dataset_name',help='dataset to be used')
    parser.add_argument('--generate',help='1 if you want to generate answers')
    parser.add_argument('--API_TOKEN',help='API token for the model to be used')
    args = parser.parse_args()
    file_name = args.dataset_name + "_" + args.model_name + "_trial.json"
    
    if os.path.isfile(file_name) and args.generate != "1":
        gold_and_predictions(args.dataset_name, args.model_name)
    else:
        generate_answers(args.dataset_name, args.model_name, args.API_TOKEN)
        #gold_and_predictions(args.dataset_name, args.model_name)