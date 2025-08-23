import numpy as np
import argparse
import torch
import pickle
import itertools
import os
import random
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, BertModel
from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run offline vLLM inference on medical code pairs using batch generation."
    )
    parser.add_argument("--model_id", type=str, default="",
                        help="The model identifier or path to the model directory (Llama-3.1-8B-Instruct).")
    parser.add_argument("--embedding_model", type=str, default="",
                        help="The model identifier or path to the model directory (bert-base-uncased).")
    parser.add_argument("--n_sample", type=int, default=1,
                        help="Number of responses to generate per prompt.")
    parser.add_argument("--max_tokens", type=int, default=128,
                        help="Maximum number of tokens to generate for each prompt.")
    parser.add_argument("--tp", type=int, default=8,
                        help="Number of tensor parallelism (TP) to use during generation.")
    parser.add_argument("--temperature", type=float, default=0.,
                        help="Sampling temperature to use during generation.")
    parser.add_argument("--code_pickle", type=str, default="./gen_llm_relation/nodes.pickle",
                        help="Path to the pickle file containing medical codes.")
    parser.add_argument("--num_codes", type=int, default=1000,
                        help="Number of codes to randomly sample from the pickle file.")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory where the generated responses and embeddings will be saved.")
    parser.add_argument("--chunk_size", type=int, default=50000,
                    help="Number of prompts to generate per chunk. Helps reduce memory usage.")

    return parser.parse_args()


def read_pickle_file(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def save_response_to_file(response: str, prompt_idx: int, sample_idx: int, output_dir: str):
    """Save a generated response to file with a unique prompt/sample index."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f'response_{prompt_idx}_{sample_idx}.txt')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(response)


def load_all_responses(output_dir: str):
    """Load all saved responses from the specified output directory."""
    responses = []
    for filename in os.listdir(output_dir):
        if filename.startswith('response_') and filename.endswith('.txt'):
            with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                responses.append(f.read())
    return responses


def get_BERT_embeddings(strings: list, model_name='bert-base-uncased'):
    """Compute and return the [CLS] token embeddings using a BERT model."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(strings, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding


def main():
    args = parse_arguments()
    random.seed(0)
    torch.manual_seed(0)

    print("Loading codes...")
    code_list = np.array(read_pickle_file(args.code_pickle))
    try:
        code_list = code_list[:, 1]
    except IndexError:
        pass
    if args.num_codes >= len(code_list):
        code_list = code_list.tolist()
        args.num_codes = len(code_list)
        print("Using all codes.")
    else:
        code_list = random.sample(code_list.tolist(), args.num_codes)
    code_pairs = list(itertools.permutations(code_list, 2))
    print(f"Sampled {len(code_pairs)} code pairs.")
    output_dir = os.path.join(
    args.output_dir,
    f"{args.num_codes}code_{args.model_id.split('/')[-1]}_temp_{args.temperature}_n{args.n_sample}_responses/batch"
)
    os.makedirs(output_dir, exist_ok=True)
    # Build all prompts first
    all_prompts = []
    for pair in tqdm(code_pairs, desc="Building prompts"):
        code_1, code_2 = pair
        prompt = f'''
### Instruction:

Given a pair of medical codes (which could refer to a diagnosis, procedure, or drug), extrapolate the most plausible relationships between the codes.
The relationships should be useful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction, etc.).
Each response should be exactly in the format [{{ENTITY 1}}, {{RELATIONSHIP}}, {{ENTITY 2}}] where the relationship is directed. NOTHING ELSE SHOULD BE GENERATED.
ENTITY 1 can be either code 1 or code 2 and ENTITY 2 can be either code 1 or code 2, provided they are different.
Each RELATIONSHIP must be conclusive and kept as short as possible.
Curly brackets {{}} must encapsulate each entity or relationship produced.
If the relationship between the codes is largely indirect, output the entities with RELATIONSHIP = N/A like this: [{{ENTITY 1}}, {{N/A}}, {{ENTITY 2}}].

Example 1:
code pair: ['OPIOID ANALGESICS', 'ASPIRIN']
response: [{{OPIOID ANALGESICS}}, {{HAS SIMILAR EFFECT AS}}, {{ASPIRIN}}]

Example 2:
code pair: ['Osteoporosis', 'Diverticulosis and diverticulitis']
response: [{{Osteoporosis}}, {{N/A}}, {{Diverticulosis and diverticulitis}}]

Prompt:
code 1: {code_1}
code 2: {code_2}

### Response:'''
        all_prompts.append([{"role": "user", "content": prompt}])
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if not tokenizer.chat_template:
        raise ValueError("The model does not have a chat template.")
    llm_engine = LLM(args.model_id,
                     dtype="bfloat16",
                     tensor_parallel_size=args.tp,
                     tokenizer=args.model_id,
                     seed=0)

    sampling_params = SamplingParams(
        n=args.n_sample,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    chunk_size = args.chunk_size
    total_chunks = (len(all_prompts) + chunk_size - 1) // chunk_size

    print(f"Generating in {total_chunks} chunks of size {chunk_size}...")

    for chunk_idx in range(total_chunks):
        chunk_prompts = all_prompts[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        print(f">>> Generating chunk {chunk_idx + 1}/{total_chunks} with {len(chunk_prompts)} prompts")

        results = llm_engine.chat(chunk_prompts, sampling_params)

        for prompt_offset, request_output in enumerate(results):
            prompt_idx = chunk_idx * chunk_size + prompt_offset
            for sample_idx, output in enumerate(request_output.outputs):
                response_text = output.text
                save_response_to_file(response_text, prompt_idx, sample_idx, output_dir)

    # After all chunks are generated, compute embeddings
    print("Loading all responses for BERT embedding...")
    all_responses = load_all_responses(output_dir)
    embeds = get_BERT_embeddings(all_responses, args.embedding_model)
    np.save(os.path.join(output_dir, 'all_run_embeddings.npy'), embeds)


if __name__ == "__main__":
    main()
