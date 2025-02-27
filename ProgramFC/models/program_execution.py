import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
from tqdm import tqdm
import re
from ProgramFC.models.question_answering import T5_Question_Answering
#from ProgramFC.models.retriever import PyseriniRetriever

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--FV_data_path', type=str)
    parser.add_argument('--setting', help='[gold | open-book | close-book]', type=str)
    parser.add_argument('--num_eval_samples', default=2000, type=int)
    parser.add_argument('--program_dir', type=str)
    parser.add_argument('--program_file_name', type=str)
    parser.add_argument('--output_dir', type=str)
    # fact checker args
    parser.add_argument("--model_name", default = 'google/flan-t5-xl', type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument('--corpus_index_path', default=None, type=str)
    parser.add_argument('--num_retrieved', default=5, type=int)
    parser.add_argument('--max_evidence_length', default=3000, help = 'to avoid exceeding GPU memory', type=int)
    args = parser.parse_args()
    return args

class Program_Execution:
    def __init__(self) -> None:
        # load model
        self.model_name = 'google/flan-t5-xl'
        self.dataset_name = 'HOVER'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        # 移除parallelize调用，因为我们使用DDP进行分布式训练
        self.setting = 'gold'
        self.QA_module = T5_Question_Answering(self.model, self.tokenizer)
        self.corpus_index_path = './datasets/HOVER/corpus/index'
        # load retriever
        """if self.setting == 'open-book':
            # 确保Java环境变量设置正确
            import os
            java_home = os.environ.get('JAVA_HOME')
            if not java_home:
                os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'  # 设置默认Java路径
            self.searcher = PyseriniRetriever(self.corpus_index_path, use_bm25=True, k1=0.9, b=0.4)
        else:
            self.searcher = None"""
        self.searcher = None
        self.sample = None

        # load dataset
        self.gold_evidence_map = None

    def map_direct_answer_to_label(self, predict):
        predict = predict.lower().strip()
        label_map = {'true': True, 'false': False, 'yes': True, 'no': False, "it's impossible to say": False}
        if predict in label_map:
            return label_map[predict]
        else:
            print(f"Alert!!! wrong answer mapping: {predict}")
            return random.sample([True, False], 1)[0]

    def parse_verify_command(self, command, variable_map):
        return_var, tmp = command.split('= Verify')
        return_var = return_var.strip()
        # claim = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Verify\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        claim = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if claim.find(replace_var) >=0:
                claim = claim.replace(replace_var, variable_value)

        return return_var, claim

    def parse_question_command(self, command, variable_map):
        return_var, tmp = command.split('= Question')
        return_var = return_var.strip()
        # question = tmp.replace("\")", "").strip()

        p1 = re.compile(f'Question\([f]?\"(.*)\"\)', re.S)
        matching = re.findall(p1, command)
        question = matching[0] if len(matching)>0 else tmp

        # replace variable
        for variable_name, variable_value in variable_map.items():
            replace_var = "{" + str(variable_name) + "}"
            if question.find(replace_var) >=0:
                question = question.replace(replace_var, variable_value)

        return return_var, question

    def get_command_type(self, command):
        if command.find("label = ")>=0:
            return "FINAL"
        elif command.find('= Verify')>=0:
            return "VERIFY"
        elif command.find('= Question')>=0:
            return "QUESTION"
        else:
            return "UNKNOWN"

    def derive_final_answer(self, command, variable_map):
        final_label = True
        command = command.replace('label =', '').strip()
        p1 = re.compile(r'Predict[(](.*?)[)]', re.S)
        command_arg = re.findall(p1, command)[0]
        verify_subs = command_arg.split(" and ")
        arguments = [arg.strip() for arg in verify_subs]
        for argument in arguments:
            if argument in variable_map:
                final_label = variable_map[argument] and final_label
            else:
                print(f"Alert!!! wrong argument: {argument}")
        return final_label

    def retrieve_evidence(self, query):
        hits = self.searcher.retrieve(query, self.args.num_retrieved)
        evidence = '\n'.join([hit['text'].strip() for hit in hits])
        # cut overlong evidence
        if len(evidence.split()) > self.args.max_evidence_length:
            print('evidence is too long, cut it to max_evidence_length')
            evidence = ' '.join(evidence.split()[:self.args.max_evidence_length])
        
        # save retrieval results (can comment out if not needed)
        retrieved_results = []
        for hit in hits:
            retrieved_results.append({'id': hit['doc_id'], 'score': hit['score'], 'query': query})
        
        return evidence, retrieved_results
    
    def parse_program(self, ID, program, evidence):
        variable_map = {}
        claim_only = True if self.args.setting == 'close-book' else False
        retrieved_evidence = []
        # for each command
        for command in program:
            c_type = self.get_command_type(command)
            final_answer = None
            # verify a claim
            if c_type == "VERIFY":
                return_var, claim = self.parse_verify_command(command, variable_map)
                # if open-book setting, then retrieve evidence from the corpus
                if self.args.setting == 'open-book':
                    evidence, retrieved_results = self.retrieve_evidence(claim)
                    retrieved_evidence += retrieved_results
                
                answer = self.QA_module.answer_verify_question(claim, evidence, claim_only)['answer_text']
                variable_map[return_var] = self.map_direct_answer_to_label(answer)
            # ask a question
            elif c_type == "QUESTION":
                return_var, question = self.parse_question_command(command, variable_map)
                # if open-book setting, then retrieve evidence from the corpus
                if self.args.setting == 'open-book':
                    evidence, retrieved_results = self.retrieve_evidence(question)
                    retrieved_evidence += retrieved_results
                
                answer = self.QA_module.answer_question_directly(question, evidence, claim_only)['answer_text']
                variable_map[return_var] = answer
            elif c_type == 'FINAL':
                try:
                    final_answer = self.derive_final_answer(command, variable_map)
                except:
                    print(f"Alert!!! parsing error: {ID}")
                    final_answer = random.sample([True, False], 1)[0]
        
        return final_answer, retrieved_evidence

    def execute_on_dataset(self, sample):
        self.sample = sample
        self.gold_evidence_map = {sample['id']:sample['evidence']}
        dataset = self.sample

        gt_labels, predictions = [], []
        results = []
        for sample in tqdm(dataset):
            program = sample['predicted_programs']
            gt_labels.append(sample['gold'])
            # get evidence
            evidence = self.gold_evidence_map[sample['id']] if self.args.setting == 'gold' else None
            
            # execute program
            sample_predictions = []
            for sample_program in program:
                try:
                    single_prediction, retrieved_evidence = self.parse_program(sample['id'], sample_program, evidence)
                except Exception as e:
                    print(f"Alert!!! execution error: {sample['id']}")
                    single_prediction = random.sample([True, False], 1)[0]
                sample_predictions.append(single_prediction)
            
            true_count = len([pred for pred in sample_predictions if pred == True])
            false_count = len([pred for pred in sample_predictions if pred == False])
            final_prediction = True if true_count > false_count else False
            predictions.append('supports' if final_prediction == True else 'refutes')
            results.append({'id': sample['id'], 
                            'claim': sample['claim'],
                            'gold': sample['gold'], 
                            'prediction': 'supports' if final_prediction == True else 'refutes'})
            print(final_prediction)
            return final_prediction

if __name__ == "__main__":
    args = parse_args()
    program_executor = Program_Execution(args)
    program_executor.execute_on_dataset()