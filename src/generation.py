"""
generation.py
-------------
STEP 4 of the pipeline: Response Generation.

Loads Llama-2-7b-chat-hf and generates an answer conditioned on the
retrieved context (from Step 3) + the user's question.

WHY THIS COUNTS AS "PROMPT ENGINEERING" NOT "FINE-TUNING" (viva point):
The brief mentions both. This project uses prompt engineering: we don't
change any model weights. Instead we wrap every query in a system prompt
(config.LLAMA2_CHAT_PROMPT_TEMPLATE) that (a) restricts the model to the
provided context and (b) instructs it to say "I don't know" rather than
hallucinate. Fine-tuning would mean further training Llama-2's weights on
a custom dataset -- far more expensive in compute/data and unnecessary for
a document-QA use case, since the context injection already does the job.

WHY A FALLBACK MODEL EXISTS (viva point):
meta-llama/Llama-2-7b-chat-hf is "gated" on Hugging Face: you must request
access from Meta and log in with an approved HF token, and even then it
needs a GPU with ~13GB VRAM (fp16) or ~6GB (4-bit). Many student machines
/ free Colab tiers don't have that headroom. To keep the app runnable and
demoable everywhere, generation.py tries to load Llama-2 first and falls
back to a small ungated model (TinyLlama-1.1B-Chat) if that fails. The
RAG pipeline (retrieval, prompting, evaluation) is IDENTICAL either way --
only the generator model changes, which is exactly the kind of modularity
a RAG architecture is supposed to give you.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from . import config


class Generator:
    def __init__(self, use_4bit: bool = True, hf_token: str = None):
        self.model = None
        self.tokenizer = None
        self.model_name_loaded = None
        self._load(use_4bit=use_4bit, hf_token=hf_token)

    def _load(self, use_4bit: bool, hf_token: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        quant_config = None
        if use_4bit and device == "cuda":
            # 4-bit quantization (via bitsandbytes) shrinks Llama-2-7b from
            # ~13GB to ~4-6GB of VRAM, at a small cost to output quality.
            # This is what makes Llama-2 runnable on a single consumer GPU
            # (e.g. an RTX 3060) instead of needing a data-center card.
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.LLAMA2_MODEL_NAME, token=hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                config.LLAMA2_MODEL_NAME,
                token=hf_token,
                quantization_config=quant_config,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.model_name_loaded = config.LLAMA2_MODEL_NAME
        except Exception as e:
            print(f"[generation.py] Could not load Llama-2 ({e}). "
                  f"Falling back to {config.FALLBACK_MODEL_NAME}.")
            self.tokenizer = AutoTokenizer.from_pretrained(config.FALLBACK_MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.FALLBACK_MODEL_NAME,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self.model_name_loaded = config.FALLBACK_MODEL_NAME

        if device == "cpu":
            self.model.to("cpu")

    def generate(self, question: str, context: str) -> str:
        """
        Builds the RAG prompt (context + question) and generates an answer.
        This is the step that turns "retrieval" into "retrieval-AUGMENTED
        GENERATION": the LLM never sees the question alone, it always sees
        it wrapped with the evidence retrieved in Step 3.
        """
        if not context.strip():
            prompt_context = "No relevant context was found in the uploaded documents."
        else:
            prompt_context = context

        prompt = config.LLAMA2_CHAT_PROMPT_TEMPLATE.format(
            context=prompt_context, question=question
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                do_sample=config.TEMPERATURE > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip the prompt back off so we only return the newly generated text.
        answer = full_output[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        return answer.strip()
