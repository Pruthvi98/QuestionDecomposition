import json
import requests
import csv
import argparse
import os
import time
import torch
import openai
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import re

from word2number import w2n

with open("data/prompts.json", "r") as file:
    prompts = json.load(file)

device = 0 if torch.cuda.is_available() else -1

def open_file(file_name):
  file = open(file_name, "r")
  data = json.load(file)
  file.close()

  return data

def replace_hash_qasc(old, st):
    
    return old.replace("#", st)

def replace_hash(old, decomp_list):
    
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
    print(args)
    if args[0] == 'summation' or args[0] =='addition':
        ans = sum_answer(args[1:])
    elif args[0] == 'difference' or args[0] == "subtraction":
        ans = diff_answer(args[1:])
    elif args[0] == "greater":
        ans = greater_answer(args[1:])
    elif args[0] == "date":
        ans = date_answer(args[1:])
    elif args[0] == "lesser":
        ans = lesser_answer(args[1:])
    elif args[0] == "division":
        ans = division_answer(args[1:])
    elif args[0] == "multiplication":
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
    ans_sum = 0
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
        print(args[i])
        ans_sum = ans_sum + args[i]
    return str(ans_sum)  
        
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
    ans_diff = args[0]
    for i in args[1:]:
        ans_diff = ans_diff - i
            
    return str(ans_diff)

def date_answer(args):
    try:
        ans_list = []
        for i in range(len(args)):
            dmy = args[i].split()
            temp = ""
            for j in dmy:
                if len(j) == 2:
                    temp += " %d "
                elif len(j) == 4:
                    temp += " %Y "
                else:
                    temp += " %B "
            
            temp = temp.strip()
            
            try:
                datetime_object = datetime.strptime(args[i], temp)
                ans_list.append(datetime_object)
            except Exception:
                datetime_object = 0
                ans_list.append(datetime_object)
        
        ans = min(ans_list)
        
        ans = ans.strftime("%B %Y")
        
        return ans
    except Exception:
        return 0
    

    
    
def get_answers_gpt3(args, context, question, true_answer = "", original_boolean = False):
    
    if args.dataset_name == "qasc":
        if original_boolean:
            prompt = prompts["qasc"]
        else:
            prompt = prompts["qasc_decomposed"]
        response = openai.Completion.create(
                                            engine="text-davinci-002",
                                            prompt=prompt.format(question),
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                           )
    else:
        prompt = prompts[args.dataset_name]
        response = openai.Completion.create(
                                            engine="text-davinci-002",
                                            prompt=prompt.format(context,question),
                                            temperature=0,
                                            max_tokens=256,
                                            top_p=1,
                                            frequency_penalty=0,
                                            presence_penalty=0
                                           )
    
    choices = response.get('choices',0)
    if choices != 0:
        answer = choices[0]['text'].strip()
        i = answer.find(":")
        
        if i != -1:
            answer = str(answer[i+1:])
    else:
        answer = choices
        
    return answer
        

def get_answers_all_gpt3(args, data, dataset_name):
    
    true_answers = []
    original_preds = []
    true_answers_decomp = []
    decomp_preds = []
    p = 0
    for data_point in data:
        time.sleep(2)

        answer = get_answers_gpt3(args, data_point['context'], data_point["question"] , data_point["answer"], True)
        
        
        data_point["generated"] = answer
        true_answers.append(data_point["answer"])
        original_preds.append(answer)
        
        
        file = open("data/" + dataset_name + "_trial_new.json", "w")
        json.dump(data, file, indent = 4)
        file.close()
        
        
        answer = answer.strip().lower()
        if answer == data_point["answer"].lower() or answer in data_point["answer"].lower():
            continue
        
        decomp_list = []
        
        if p >= 50:
            break
        
        i = 0

        if args.dataset_name == "qasc":
            use_question = data_point["decompositions"][0]["question"]
            decomp_list = []
            for item in data_point['choices']:
                
                new_question = replace_hash_qasc(use_question, item['text'])
                
                
                time.sleep(2)
                answer = get_answers_gpt3(args, data_point['context'], new_question)
                answer = answer.strip().lower()
                print((new_question, answer))
                data_point["new_question"].append({"question": new_question, "answer": answer})
                
                if "yes" in answer:
                    decomp_list.append(item['text'])
                
                i += 1
                
            

            answer = answer.strip().lower()
            true_answers_decomp.append(data_point["answer"].lower())
            decomp_preds.append(decomp_list)
            file = open("data/" + dataset_name + "_trial_new.json", "w")
            json.dump(data, file, indent = 4)
            file.close()
        else:
            l = len(data_point["decompositions"]) - 1

            for item in data_point['decompositions']:
                if "#" in item["question"]:
                    item["question"] = replace_hash(item["question"], decomp_list)
                
                if "!" in item["question"]:
                    answer = calculate(item["question"])
                
                else:
                    
                    time.sleep(2)
                    answer = get_answers_gpt3(args, data_point['context'], item['question'])
                    
                i += 1
                decomp_list.append(answer)
                item["generated"] = answer
                print(data_point)
                
                if i == l+1:
                    print((data_point["answer"], data_point["generated"]))
                    print((data_point["answer"], answer))
                    answer = answer.strip().lower()
                    true_answers_decomp.append(data_point["answer"].lower())
                    decomp_preds.append(answer)
                file = open("data/" + dataset_name + "_trial_new.json", "w")
                json.dump(data, file, indent = 4)
                file.close()
        p += 1
        
        file = open("data/" + dataset_name + "_original.json", "w")
        original = {"true": true_answers, "prediction": original_preds}
        json.dump(original, file, indent = 4)
        file.close()
        
        file = open("data/" + dataset_name + "_decomposition.json", "w")
        original = {"true": true_answers_decomp, "prediction": decomp_preds}
        json.dump(original, file, indent = 4)
        file.close()


def generate_answers(args, dataset_name, model_name, API_TOKEN):
    file = "data/" + dataset_name + ".json"
    
    data = open_file(file)
    if model_name == "roberta":
        responses = get_answers_roberta(data, dataset_name)
    
    if model_name == "gpt3":
        openai.api_key = API_TOKEN
        responses = get_answers_all_gpt3(args, data, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation of Question Decomposition')
    parser.add_argument('--model_name', help='model to be used')
    parser.add_argument('--dataset_name',help='dataset to be used')
    parser.add_argument('--API_TOKEN',help='API token for the model to be used')
    args = parser.parse_args()
    
    
    generate_answers(args, args.dataset_name, args.model_name, args.API_TOKEN)