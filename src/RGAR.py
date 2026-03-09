# This file is adapted from [Teddy-XiongGZ/MedRAG]  
# Original source: [https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/medrag.py]  
# we developed RGAR based on MedRAG
# we developed RAG systems without CoT based on MedRAG
# we add support for qwens
import os
import re
import json
import ast
import torch
import transformers
from transformers import AutoTokenizer
import openai
from transformers import StoppingCriteria, StoppingCriteriaList
import tiktoken
try:
    from .utils import RetrievalSystem, DocExtracter
    from .template import (
        general_cot_system,
        general_cot,
        general_medrag_system,
        general_medrag,
        general_cot_system2,
        general_cot2,
        general_medrag_system2,
        general_medrag2,
        general_extract_nolist,
        meditron_cot,
        meditron_medrag,
        simple_medrag_system,
        simple_medrag_prompt,
        i_medrag_system,
        follow_up_instruction_ask,
        follow_up_instruction_answer,
    )
    from .config import config
except ImportError:
    from utils import RetrievalSystem, DocExtracter
    from template import (
        general_cot_system,
        general_cot,
        general_medrag_system,
        general_medrag,
        general_cot_system2,
        general_cot2,
        general_medrag_system2,
        general_medrag2,
        general_extract_nolist,
        meditron_cot,
        meditron_medrag,
        simple_medrag_system,
        simple_medrag_prompt,
        i_medrag_system,
        follow_up_instruction_ask,
        follow_up_instruction_answer,
    )
    from config import config



openai.api_type = openai.api_type or os.getenv("OPENAI_API_TYPE") or config.get("api_type")
openai.api_version = openai.api_version or os.getenv("OPENAI_API_VERSION") or config.get("api_version")
openai.api_key = openai.api_key or os.getenv('OPENAI_API_KEY') or config["api_key"]

if openai.__version__.startswith("0"):
    openai.api_base = openai.api_base or os.getenv("OPENAI_API_BASE") or config.get("api_base")
    if openai.api_type == "azure":
        openai_client = lambda **x: openai.ChatCompletion.create(**{'engine' if k == 'model' else k: v for k, v in x.items()})["choices"][0]["message"]["content"]
    else:
        openai_client = lambda **x: openai.ChatCompletion.create(**x)["choices"][0]["message"]["content"]
else:
    if openai.api_type == "azure":
        openai.azure_endpoint = openai.azure_endpoint or os.getenv("OPENAI_ENDPOINT") or config.get("azure_endpoint")
        openai_client = lambda **x: openai.AzureOpenAI(
            api_version=openai.api_version,
            azure_endpoint=openai.azure_endpoint,
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content
    else:
        openai_client = lambda **x: openai.OpenAI(
            api_key=openai.api_key,
        ).chat.completions.create(**x).choices[0].message.content

class RGAR:
    RETRIEVAL_MODE_DIRECT = "direct"
    RETRIEVAL_MODE_GAR = "gar"
    RETRIEVAL_MODE_RGAR = "rgar"
    RETRIEVAL_MODE_ITERATIVE_RGAR = "iterative_rgar"
    RETRIEVAL_MODE_NAMES = (
        RETRIEVAL_MODE_DIRECT,
        RETRIEVAL_MODE_GAR,
        RETRIEVAL_MODE_RGAR,
        RETRIEVAL_MODE_ITERATIVE_RGAR,
    )

    def __init__(
        self,
        llm_name="OpenAI/gpt-3.5-turbo-16k",
        rag=True,
        follow_up=False,
        retriever_name="MedCPT",
        corpus_name="Textbooks",
        db_dir="./corpus",
        cache_dir=None,
        corpus_cache=False,
        HNSW=False,
        device="auto",
        cot=False,
        retrieval_mode="direct",
        iterative_rounds=2,
        follow_up_rounds=2,
        follow_up_queries=3,
    ):
        self.llm_name = llm_name
        self.rag = rag
        self.follow_up = bool(follow_up)
        self.retrieval_mode = retrieval_mode
        if self.retrieval_mode not in self.RETRIEVAL_MODE_NAMES:
            valid_modes = ", ".join(self.RETRIEVAL_MODE_NAMES)
            raise ValueError(f"Unsupported retrieval mode: {self.retrieval_mode}. Expected one of {valid_modes}.")
        self.iterative_rounds = max(1, int(iterative_rounds))
        self.follow_up_rounds = max(1, int(follow_up_rounds))
        self.follow_up_queries = max(1, int(follow_up_queries))
        self.retriever_name = retriever_name
        self.corpus_name = corpus_name
        self.db_dir = db_dir
        self.cache_dir = cache_dir
        self.docExt = None
        if rag:
            self.retrieval_system = RetrievalSystem(self.retriever_name, self.corpus_name, self.db_dir, cache=corpus_cache, HNSW=HNSW)
        else:
            self.retrieval_system = None
        if cot:
            self.templates = {"cot_system": general_cot_system, "cot_prompt": general_cot,
                    "medrag_system": general_medrag_system, "medrag_prompt": general_medrag}
        else:
            self.templates = {"cot_system": general_cot_system2, "cot_prompt": general_cot2,
                    "medrag_system": general_medrag_system2, "medrag_prompt": general_medrag2}
        self.templates["general_extract"]=general_extract_nolist
        if self.llm_name.split('/')[0].lower() == "openai":
            self.model = self.llm_name.split('/')[-1]
            if "gpt-3.5" in self.model or "gpt-35" in self.model:
                self.max_length = 16384
                self.context_length = 15000
            elif "gpt-4" in self.model:
                self.max_length = 32768
                self.context_length = 30000
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        elif "gemini" in self.llm_name.lower():
            import google.generativeai as genai
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            self.model = genai.GenerativeModel(
                model_name=self.llm_name.split('/')[-1],
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 2048,
                }
            )
            if "1.5" in self.llm_name.lower():
                self.max_length = 1048576
                self.context_length = 1040384
            else:
                self.max_length = 30720
                self.context_length = 28672
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.max_length = 2048
            self.context_length = 1024
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
            if "mixtral" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/mistral-instruct.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 32768
                self.context_length = 30000
            elif "llama-2" in llm_name.lower():
                self.max_length = 4096
                self.context_length = 3072
            elif "llama-3" in llm_name.lower():
                self.max_length = 8192
                self.context_length = 7168
                if ".1" in llm_name or ".2" in llm_name:
                    self.max_length = 131072
                    self.context_length = 128000
            elif "meditron-70b" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/meditron.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 4096
                self.context_length = 3072
                self.templates["cot_prompt"] = meditron_cot
                self.templates["medrag_prompt"] = meditron_medrag
            elif "pmc_llama" in llm_name.lower():
                self.tokenizer.chat_template = open('./templates/pmc_llama.jinja').read().replace('    ', '').replace('\n', '')
                self.max_length = 2048
                self.context_length = 1024
            elif "qwen" in llm_name.lower():
                self.max_length = 131072
                self.context_length = 128000
                
            self.model = transformers.pipeline(
                "text-generation",
                model=self.llm_name,
                # torch_dtype=torch.float16,
                torch_dtype=torch.bfloat16,
                device_map=device,
                model_kwargs={"cache_dir":self.cache_dir},
            )
            if "llama-3" in llm_name.lower():
                self.tokenizer=self.model.tokenizer

        if self.rag and self.follow_up:
            self.templates["medrag_system"] = simple_medrag_system
            self.templates["medrag_prompt"] = simple_medrag_prompt
            self.templates["i_medrag_system"] = i_medrag_system
            self.templates["follow_up_ask"] = follow_up_instruction_ask
            self.templates["follow_up_answer"] = follow_up_instruction_answer

    def answer(self, *args, **kwargs):
        if self.rag and self.follow_up:
            return self.i_medrag_answer(*args, **kwargs)
        return self.medrag_answer(*args, **kwargs)

    @staticmethod
    def _format_options(options):
        if options is None:
            return ""
        if isinstance(options, dict):
            return '\n'.join([key + ". " + options[key] for key in sorted(options.keys())])
        return str(options)

    @staticmethod
    def _join_query_parts(*parts):
        valid_parts = []
        for part in parts:
            if part is None:
                continue
            part_str = str(part).strip()
            if part_str:
                valid_parts.append(part_str)
        return "\n".join(valid_parts)

    @staticmethod
    def _split_budget(total, parts):
        if parts <= 0:
            return []
        base = total // parts
        extra = total % parts
        return [base + (1 if i < extra else 0) for i in range(parts)]

    def _run_multi_query_retrieval(self, queries, budgets, rrf_k=100):
        all_retrieved_snippets = []
        all_scores = []
        for query, budget in zip(queries, budgets):
            if budget <= 0:
                continue
            cleaned_query = str(query).strip()
            if not cleaned_query:
                continue
            retrieved_snippets, scores = self.retrieval_system.retrieve(cleaned_query, k=budget, rrf_k=rrf_k)
            all_retrieved_snippets.extend(retrieved_snippets)
            all_scores.extend(scores)
        return all_retrieved_snippets, all_scores

    def _build_contexts(self, retrieved_snippets):
        contexts = [
            "Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"])
            for idx in range(len(retrieved_snippets))
        ]
        if len(contexts) == 0:
            return [""]
        context_text = "\n".join(contexts)
        if "openai" in self.llm_name.lower() or "gemini" in self.llm_name.lower():
            return [self.tokenizer.decode(self.tokenizer.encode(context_text)[:self.context_length])]
        return [self.tokenizer.decode(self.tokenizer.encode(context_text, add_special_tokens=False)[:self.context_length])]

    @staticmethod
    def _safe_parse_list(raw_text):
        if raw_text is None:
            return None
        text = str(raw_text).strip()
        if text == "":
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return parsed
        return None

    def _select_retrieval_strategy(self):
        strategies = {
            self.RETRIEVAL_MODE_DIRECT: lambda question, options, k, rrf_k: self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k),
            self.RETRIEVAL_MODE_GAR: self.retrieve_with_gar,
            self.RETRIEVAL_MODE_RGAR: self.retrieve_with_rgar,
            self.RETRIEVAL_MODE_ITERATIVE_RGAR: self.retrieve_with_iterative_rgar,
        }
        return strategies[self.retrieval_mode]

    def custom_stop(self, stop_str, input_len=0):
        stopping_criteria = StoppingCriteriaList([CustomStoppingCriteria(stop_str, self.tokenizer, input_len)])
        return stopping_criteria

    def generate(self, messages, **kwargs):
        '''
        generate response given messages
        '''
        if "openai" in self.llm_name.lower():
            ans = openai_client(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
        elif "gemini" in self.llm_name.lower():
            response = self.model.generate_content(messages[0]["content"] + '\n\n' + messages[1]["content"])
            ans = response.candidates[0].content.parts[0].text
        else:
            stopping_criteria = None
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if "meditron" in self.llm_name.lower():
                # stopping_criteria = custom_stop(["###", "User:", "\n\n\n"], self.tokenizer, input_len=len(self.tokenizer.encode(prompt_cot, add_special_tokens=True)))
                stopping_criteria = self.custom_stop(["###", "User:", "\n\n\n"], input_len=len(self.tokenizer.encode(prompt, add_special_tokens=True)))
            elif "llama-3" in self.llm_name.lower():
                response = self.model(
                        prompt,
                        temperature=None, 
                        top_p=None,  
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        # max_length=self.max_length,
                        max_new_tokens=4096,
                        repetition_penalty=1.2,
                        truncation=True,
                        stopping_criteria=None,
                )
            elif "qwen" in self.llm_name.lower():
                response = self.model(
                        prompt,
                        temperature=None,  
                        top_p=None,  
                        do_sample=False,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        # max_length=self.max_length,
                        max_new_tokens=4096,
                        repetition_penalty=1.2,
                        truncation=True,
                        stopping_criteria=None,
                ) 
            else:
                response = self.model(
                    prompt,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_length,
                    truncation=True,
                    stopping_criteria=stopping_criteria
                )
            # ans = response[0]["generated_text"]
            ans = response[0]["generated_text"][len(prompt):]
        return ans
    def extract_factual_info_rag(self,question,retrieved_snippets):
        num_sentences, other_sentences, last_sentence = self.split_sentences(question)
        contexts = ["Document [{:d}] (Title: {:s}) {:s}".format(idx, retrieved_snippets[idx]["title"], retrieved_snippets[idx]["content"]) for idx in range(len(retrieved_snippets))]
        answers = []
        if len(contexts) == 0:
            contexts = [""]
        if "openai" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
        elif "gemini" in self.llm_name.lower():
            contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts))[:self.context_length])]
        else:
            contexts = [self.tokenizer.decode(self.tokenizer.encode("\n".join(contexts), add_special_tokens=False)[:self.context_length])]
        for context in contexts:
            
            prompt_extract = self.templates["general_extract"].render(context=context, ehr=other_sentences, question=last_sentence)
            messages=[
                    
                    {"role": "user", "content": prompt_extract}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        return answers  
    def extract_factual_info(self,question):
        # prompt = '''Please extract the key factual information relevant to solving this problem and present it as a Python list. 
        # Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"].'''
        prompt = '''Please extract the key factual information relevant to solving this problem and present it as a Python list. 
        Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"]. For example, ['Patient age: 39 years (Middle-aged)', 'Symptoms: fever, chills, left lower quadrant abdominal pain', 'Vital signs: high temperature (39.1°C or 102.3°F), tachycardia (pulse 126/min), tachypnea (respirations 28/min) and hypotension (blood pressure 80/50 mmHg)', 'Physical exam findings: mucopurulent discharge from the cervical os and left adnexal tenderness', 'Laboratory results: low platelet count (14,200/mm^3), elevated D-dimer (965 ng/mL)', 'Phenol test result: identification of a phosphorylated N-acetylglucosame dimmer with 6 fatty acids attached to a polysaccharide side chain']'''
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]
        
        matched_items = re.findall(r'"([^"]*)"', answers)

        if matched_items:
            return matched_items,answers
        else:
            return [],answers
    def generate_possible_content(self,question):

        prompt = '''Please generate some knowledge that might address the above question. please give me only the knowledge.'''

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]
        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers
    def generate_possible_answer(self,question):

        # prompt = '''Please generate some knowledge that might address the above question. please give me only the knowledge.'''
        prompt = '''Please give 4 options for the question. Each option should be a concise description of a key detail, formatted as:A. "key detail 1" B. "key detail 2" C. "key detail 3" D. "key detail 4"'''
        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers
    def generate_possible_title(self,question):

        prompt = '''Please generate some titles of references that might address the above question. Please give me only the titles, formatted as: ["title 1", "title 2", ..., "title N"]. Please be careful not to give specific content and analysis, just the title.'''

        messages = [
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": question + "\n" + prompt},
        ]

        ans = self.generate(messages)
        answers = []
        answers.append(re.sub("\s+", " ", ans))
        answers = answers[0]

        # print(f"Generated Answer: {answers}")
        return answers
    def split_sentences(self,text):

        text = text.rstrip('"').strip()

        pattern = r'(.*?[.!?。\n])'  
        sentences = re.findall(pattern, text, re.DOTALL)  

        if not sentences:  
            return 0, "", ""

        last_sentence = sentences[-1].strip()
        other_sentences = "".join(sentences[:-1]).strip()  
        
        return len(sentences), other_sentences, last_sentence

    def retrieve_with_rgar(self, question, options="", k=32, rrf_k=100):
        """
        RGAR retrieval:
        1) Extract factual details from EHR/question text when available.
        2) Use the facts to seed GAR-style query generation.
        """
        _, other_sentences, last_sentence = self.split_sentences(question)
        factual_seed = ""
        if other_sentences != "":
            factual_items, factual_raw = self.extract_factual_info(question)
            factual_seed = "; ".join(factual_items) if factual_items else factual_raw

        query_seed = self._join_query_parts(factual_seed, last_sentence) or question
        option_text = self._format_options(options)
        possible_answers = self.generate_possible_answer(query_seed)
        possible_content = self.generate_possible_content(query_seed)
        possible_title = self.generate_possible_title(query_seed)

        queries = [
            self._join_query_parts(query_seed, option_text, possible_answers),
            self._join_query_parts(query_seed, possible_content),
            self._join_query_parts(query_seed, possible_title),
        ]
        budgets = self._split_budget(k, len(queries))
        return self._run_multi_query_retrieval(queries, budgets, rrf_k=rrf_k)
    
    def retrieve_with_gar(self, question, options="", k=32, rrf_k=100):
        """
        GAR retrieval without explicit factual extraction.
        """
        _, _, last_sentence = self.split_sentences(question)
        option_text = self._format_options(options)
        possible_content = self.generate_possible_content(question)
        possible_title = self.generate_possible_title(question)

        queries = [
            last_sentence or question,
            option_text,
            possible_content,
            possible_title,
        ]
        budgets = self._split_budget(k, len(queries))
        return self._run_multi_query_retrieval(queries, budgets, rrf_k=rrf_k)

    def retrieve_with_iterative_rgar(self, question, options="", k=32, rrf_k=100):
        """
        Multi-round RGAR retrieval:
        repeatedly extract facts from current retrieval results, then regenerate
        retrieval queries for the next round.
        """
        option_text = self._format_options(options)
        retrieved_snippets, scores = self.retrieval_system.retrieve(question, k=k, rrf_k=rrf_k)

        for _ in range(self.iterative_rounds):
            _, other_sentences, _ = self.split_sentences(question)
            extracted_facts = ""
            if other_sentences != "" and retrieved_snippets:
                extracted_facts = self._join_query_parts(*self.extract_factual_info_rag(question, retrieved_snippets))
            query_seed = self._join_query_parts(question, extracted_facts)

            possible_answers = self.generate_possible_answer(query_seed)
            possible_content = self.generate_possible_content(query_seed)
            possible_title = self.generate_possible_title(query_seed)

            queries = [
                self._join_query_parts(query_seed, option_text, possible_answers),
                self._join_query_parts(query_seed, possible_content),
                self._join_query_parts(query_seed, possible_title),
            ]
            budgets = self._split_budget(k, len(queries))
            retrieved_snippets, scores = self._run_multi_query_retrieval(queries, budgets, rrf_k=rrf_k)
            if not retrieved_snippets:
                retrieved_snippets, scores = self.retrieval_system.retrieve(query_seed or question, k=k, rrf_k=rrf_k)

        return retrieved_snippets, scores

    def i_medrag_answer(self, question, options=None, k=32, rrf_k=100, save_path=None, n_rounds=None, n_queries=None, qa_cache_path=None, **kwargs):
        if n_rounds is None:
            n_rounds = self.follow_up_rounds
        if n_queries is None:
            n_queries = self.follow_up_queries
        n_rounds = max(1, int(n_rounds))
        n_queries = max(1, int(n_queries))

        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''
        question_prompt = f"Here is the question:\n{question}\n\n{options}"
        real_question = f"here is the question: \n{question}"
        context = ""
        qa_cache = []
        if qa_cache_path is not None and os.path.exists(qa_cache_path):
            with open(qa_cache_path, 'r', encoding='utf-8') as f:
                parsed_cache = self._safe_parse_list(f.read())
            if parsed_cache is not None:
                qa_cache = parsed_cache[:n_rounds]
                if len(qa_cache) > 0:
                    context = qa_cache[-1]
                n_rounds = n_rounds - len(qa_cache)
        last_context = None

        max_iterations = n_rounds + 3
        saved_messages = [{"role": "system", "content": self.templates["i_medrag_system"]}]

        for i in range(max_iterations):
            if i < n_rounds:
                if context == "":
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{real_question}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
                else:
                    messages = [
                        {
                            "role": "system",
                            "content": self.templates["i_medrag_system"],
                        },
                        {
                            "role": "user",
                            "content": f"{context}\n\n{real_question}\n\n{self.templates['follow_up_ask'].format(n_queries)}",
                        },
                    ]
            elif context != last_context:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{real_question}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            elif len(messages) == 1:
                messages = [
                    {
                        "role": "system",
                        "content": self.templates["i_medrag_system"],
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{real_question}\n\n{self.templates['follow_up_answer']}",
                    },
                ]
            saved_messages.append(messages[-1])
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            last_context = context
            last_content = self.generate(messages, **kwargs)
            response_message = {"role": "assistant", "content": last_content}
            saved_messages.append(response_message)
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
            if i >= n_rounds and ("## Answer" in last_content or "answer is" in last_content.lower()):
                messages.append(response_message)
                messages.append(
                    {
                        "role": "user",
                        "content": "Options:\n"+options+"\n Output the answer in JSON: {'answer': your_answer (A/B/C/D)}" if options else "Output the answer in JSON: {'answer': your_answer}",
                    }
                )
                saved_messages.append(messages[-1])
                answer_content = self.generate(messages, **kwargs)
                answer_message = {"role": "assistant", "content": answer_content}
                messages.append(answer_message)
                saved_messages.append(messages[-1])
                if save_path:
                    with open(save_path, 'w') as f:
                        json.dump([p if type(p) == dict else p.model_dump() for p in saved_messages], f, indent=4)
                return messages[-1]["content"], messages
            elif "## Queries" in last_content:
                messages = messages[:-1]
                if last_content.split("## Queries")[-1].strip() == "":
                    print("Empty queries. Continue with next iteration.")
                    continue
                try:
                    action_str = self.generate([
                        {
                            "role": "user",
                            "content": f"Parse the following passage and extract the queries as a list: {last_content}.\n\nPresent the queries as they are. DO NOT merge or break down queries. Output the list of queries in JSON format: {{\"output\": [\"query 1\", ..., \"query N\"]}}",
                        }
                    ], **kwargs)
                    action_str = re.search(r"output\": (\[.*\])", action_str, re.DOTALL).group(1)
                    parsed_actions = self._safe_parse_list(action_str)
                    if parsed_actions is None:
                        raise ValueError("Failed to parse extracted query list.")
                    action_list = [re.sub(r'^\d+\.\s*', '', str(s).strip()) for s in parsed_actions]
                except Exception as E:
                    print("Error parsing action list. Continue with next iteration.")
                    error_class = E.__class__.__name__
                    error = f"{error_class}: {str(E)}"
                    print(error)
                    if save_path:
                        with open(save_path + ".error", 'a') as f:
                            f.write(f"{error}\n")
                    continue
                for query in action_list:
                    if query.strip() == "":
                        continue
                    try:
                        rag_result = self.medrag_answer(query, k=k, rrf_k=rrf_k, **kwargs)[0]
                        context += f"\n\nQuery: {query}\nAnswer: {rag_result}"
                        context = context.strip()
                    except Exception as E:
                        error_class = E.__class__.__name__
                        error = f"{error_class}: {str(E)}"
                        print(error)
                        if save_path:
                            with open(save_path + ".error", 'a') as f:
                                f.write(f"{error}\n")
                qa_cache.append(context)
                if qa_cache_path:
                    with open(qa_cache_path, 'w') as f:
                        json.dump(qa_cache, f, indent=4)
            else:
                messages.append(response_message)
                print("No queries or answer. Continue with next iteration.")
                continue
        return messages[-1]["content"], messages

    def medrag_answer(self, question, options=None, k=32, rrf_k=100, save_dir = None, snippets=None, snippets_ids=None, **kwargs):
        '''
        question (str): question to be answered
        options (Dict[str, str]): options to be chosen from
        k (int): number of snippets to retrieve
        rrf_k (int): parameter for Reciprocal Rank Fusion
        save_dir (str): directory to save the results
        snippets (List[Dict]): list of snippets to be used
        snippets_ids (List[Dict]): list of snippet ids to be used
        '''
        
        copy_options = options
        options = self._format_options(options)

        # retrieve relevant snippets
        if self.rag:
            if snippets is not None:
                retrieved_snippets = snippets[:k]
                scores = []
            elif snippets_ids is not None:
                if self.docExt is None:
                    self.docExt = DocExtracter(db_dir=self.db_dir, cache=True, corpus_name=self.corpus_name)
                retrieved_snippets = self.docExt.extract(snippets_ids[:k])
                scores = []
            else:
                assert self.retrieval_system is not None
                retrieval_strategy = self._select_retrieval_strategy()
                retrieved_snippets, scores = retrieval_strategy(question, copy_options, k, rrf_k)
            contexts = self._build_contexts(retrieved_snippets)
        else:
            retrieved_snippets = []
            scores = []
            contexts = []

        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # generate answers
        answers = []
        if not self.rag:
            prompt_cot = self.templates["cot_prompt"].render(question=question, options=options)
            messages = [
                {"role": "system", "content": self.templates["cot_system"]},
                {"role": "user", "content": prompt_cot}
            ]
            ans = self.generate(messages)
            answers.append(re.sub("\s+", " ", ans))
        else:
            for context in contexts:
                prompt_medrag = self.templates["medrag_prompt"].render(context=context, question=question, options=options)
                messages=[
                        {"role": "system", "content": self.templates["medrag_system"]},
                        {"role": "user", "content": prompt_medrag}
                ]
                ans = self.generate(messages)
                answers.append(re.sub("\s+", " ", ans))
        
        if save_dir is not None:
            with open(os.path.join(save_dir, "snippets.json"), 'w') as f:
                json.dump(retrieved_snippets, f, indent=4)
            with open(os.path.join(save_dir, "response.json"), 'w') as f:
                json.dump(answers, f, indent=4)
        
        return answers[0] if len(answers)==1 else answers, retrieved_snippets, scores

    


class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_words, tokenizer, input_len=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stops_words = stop_words
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        tokens = self.tokenizer.decode(input_ids[0][self.input_len:])
        return any(stop in tokens for stop in self.stops_words)
