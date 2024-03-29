import os
# from safetensors.torch import load_file
from collections.abc import Sequence
from glob import glob

import numpy as np
import torch
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lmdeploy.model import MODELS, BaseModel

meta_instruction = """meta instruction
You are an AI assistant whose name is leonfrank的小秘书.
conversation
"""  # noqa


@MODELS.register_module(name='internlm-chat-7b')
class InternLMChatTemplate(BaseModel):
    """Internlm chat template."""

    def __init__(self,
                 system=meta_instruction,
                 user='<|User|>:',
                 assistant='<|Bot|>:',
                 eoh='<TOKENS_UNUSED_0>',
                 eoa='<TOKENS_UNUSED_1>',
                 stop_words=['<TOKENS_UNUSED_0>', '<TOKENS_UNUSED_1>'],
                 **kwargs):
        super().__init__(**kwargs)
        self.system = system
        self.user = user
        self.assistant = assistant
        self.eoh = eoh
        self.eoa = eoa
        self.stop_words = stop_words

    def decorate_prompt(self, prompt, sequence_start=True):
        """Apply chat template to prompt."""

        if sequence_start:
            return f'{self.system} {self.user} {prompt}{self.eoh} {self.assistant}'  # noqa
        else:
            return f' {self.user} {prompt}{self.eoh} {self.assistant}'

    def messages2prompt(self, messages, sequence_start=True):
        """Apply chat template to history."""
        if isinstance(messages, str) or isinstance(messages[0], str):
            return self.decorate_prompt(messages, sequence_start)
        system, users, assistants = self._translate_messages(messages)
        system = self.system if not system else system
        ret = system
        for user, assistant in zip(users, assistants):
            if not isinstance(user, str):
                assert isinstance(user, Sequence)
                assert all(isinstance(item, dict) for item in user)
                user = [user[0]['text'], len(user) - 1]
            if assistant:
                ret += f' {self.user} {user}{self.eoh} {self.assistant} {assistant}{self.eoa}'  # noqa
            else:
                ret += f' {self.user} {user}{self.eoh} {self.assistant}'
        return ret


class InternLMChat:
    """Internlm-chat preprocessor to prepare the inputs for a model."""

    def __init__(self, pretrained_model_name_or_path, **kwargs):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.decorator = InternLMChatTemplate(**kwargs)
        self._load_model()

    def _load_model(self):
        path = self.pretrained_model_name_or_path
        # if not os.path.exists(path):
        #     path = snapshot_download(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path,
                                                       trust_remote_code=True)
        with init_empty_weights():
            config = AutoConfig.from_pretrained(path, trust_remote_code=True)
            config.num_hidden_layers = 0  # speedup
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            model.internlm_model = None
            model.to_empty(device='cpu')
            named_parameters = set()
            for key, _ in model.named_parameters():
                named_parameters.add(key)
            # TODO: load bin according to index.json
            bins = glob(os.path.join(path, '*.bin'))
            # bins = glob(os.path.join(path, '*.safetensors'))
            for bin in bins:
                dt = torch.load(bin, map_location='cpu')
                # dt = load_file(bin)
                missed, _ = model.load_state_dict(dt, strict=False)
                named_parameters.difference_update(set(missed))
            assert len(
                named_parameters) == 0, f'missing keys: {named_parameters}'
            self.model = model.to('cuda').eval()

    @torch.no_grad()
    def encode_img(self, paths):
        """Extract image features."""
        if len(paths) == 0:
            return None
        features = []
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for path in paths:
                out = self.model.encode_img(path)
                features.append(out.squeeze().cpu().numpy())
        return features

    def _to_inputs(self, decorate_text, image_paths, sequence_start):
        features = self.encode_img(image_paths)
        input_ids = []
        ranges = None
        begins = []
        segs = decorate_text.split(self.decorator.image_placeholder)
        image_dim = features[-1].shape[0] if features is not None else 0
        for i, seg in enumerate(segs):
            if i > 0:
                begins.append(len(input_ids))
                input_ids.extend([0] * image_dim)
            seg_ids = self.tokenizer.encode(
                seg, add_special_tokens=((i == 0) and sequence_start))
            input_ids.extend(seg_ids)
        if features is not None:
            ends = np.array(begins) + image_dim
            ranges = np.stack([begins, ends], axis=1).tolist()
        return input_ids, features, ranges

    def prepare_query(self, query, sequence_start=True):
        """Convert query to input_ids, features and the ranges of features to
        input_ids."""
        image_paths = []
        if not isinstance(query, str):
            query, image_paths = query[0], query[1:]
            if len(image_paths) > 1:
                print('does not support multiple images, use last one.')
                image_paths = image_paths[-1:]
        decorate_text = self.decorator.decorate_prompt(
            (query, len(image_paths)))
        return self._to_inputs(decorate_text, image_paths, sequence_start)

    def prepare_message(self, messages):
        """Convert messages to input_ids, features and the ranges of features
        to input_ids."""
        decorate_text = self.decorator.messages2prompt(messages, True)

        return decorate_text