# coding: utf-8
# 导入必要的库
# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from contextlib import contextmanager
from lmdeploy.serve.gradio.constants import CSS, THEME, disable_btn, enable_btn
from lmdeploy.turbomind import TurboMind
from lmdeploy.turbomind.chat import valid_str
from lmdeploy.model import MODELS, BaseModel

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
MODEL_PATH='leonfrank/internlm-lmdeploy-personal_assistant'
MODEL_NAME='internlm-chat-7b'

"""A simple web interactive chat demo based on gradio."""

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


def _load_model():
    # load model
    tm_model = TurboMind.from_pretrained(MODEL_PATH, model_name='internlm-chat-7b')
    return tm_model


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def _launch_demo( tm_model, decorator):
    def prepare_query( query, sequence_start=True):
        """Convert query to input_ids, features and the ranges of features to
        input_ids."""
        decorate_text = decorator.decorate_prompt(query)

        return decorate_text

    def add_text(chatbot, session, text):
        """User query."""
        chatbot = chatbot + [(text, None)]
        history = session._message
        if len(history) == 0 or history[-1][-1] is not None:
            history.append([text, None])
        else:
            history[-1][0].insert(0, text)
        return chatbot, session, disable_btn, enable_btn

    def chat(
        chatbot,
        session,
        request_output_len=512,
    ):
        """Chat with AI assistant."""
        generator = tm_model.create_instance()
        history = session._message
        sequence_start = len(history) == 1
        seed = random.getrandbits(64) if sequence_start else None
        input_ids, features, ranges = decorator.prepare_query(
            history[-1][0], sequence_start)

        if len(input_ids
               ) + session.step + request_output_len > tm_model.model.session_len:
            gr.Warning('WARNING: exceed session max length.'
                       ' Please restart the session by reset button.')
            yield chatbot, session, enable_btn, disable_btn, enable_btn
        else:
            response_size = 0
            step = session.step
            for outputs in generator.stream_infer(
                    session_id=session.session_id,
                    input_ids=input_ids,
                    input_embeddings=features,
                    input_embedding_ranges=ranges,
                    request_output_len=request_output_len,
                    stream_output=True,
                    sequence_start=sequence_start,
                    random_seed=seed,
                    step=step):
                res, tokens = outputs[0]
                # decode res
                response = tm_model.tokenizer.decode(res.tolist(),
                                                  offset=response_size)
                if response.endswith('�'):
                    continue
                response = valid_str(response)
                response_size = tokens
                if chatbot[-1][1] is None:
                    chatbot[-1][1] = ''
                    history[-1][1] = ''
                chatbot[-1][1] += response
                history[-1][1] += response
                session._step = step + len(input_ids) + tokens
                yield chatbot, session, disable_btn, enable_btn, disable_btn

            yield chatbot, session, enable_btn, disable_btn, enable_btn

    def stop(session):
        """Stop the session."""
        generator = tm_model.create_instance()
        for _ in generator.stream_infer(session_id=session.session_id,
                                        input_ids=[0],
                                        request_output_len=0,
                                        sequence_start=False,
                                        sequence_end=False,
                                        stop=True):
            pass

    def cancel(chatbot, session):
        """Stop the session and keey chat history."""
        stop(session)
        return chatbot, session, disable_btn, enable_btn, enable_btn

    def reset(session):
        """Reset a new session."""
        stop(session)
        session._step = 0
        session._message = []
        return [], session, enable_btn

    with gr.Blocks(css=CSS, theme=THEME) as demo:
        with gr.Column(elem_id='container'):
            gr.Markdown('## LMDeploy InternLM Personal Assistant')

            chatbot = gr.Chatbot(elem_id='chatbot', label=tm_model.model_name)
            query = gr.Textbox(placeholder='Please input the instruction',
                               label='Instruction')
            session = gr.State()

        send_event = query.submit(
            add_text, [chatbot, session, query], [chatbot, session]).then(
                chat, [chatbot, session],
                [chatbot, session, query, cancel_btn, reset_btn])
        query.submit(lambda: gr.update(value=''), None, [query])

        cancel_btn.click(cancel, [chatbot, session],
                         [chatbot, session, cancel_btn, reset_btn, query],
                         cancels=[send_event])

        reset_btn.click(reset, [session], [chatbot, session, query],
                        cancels=[send_event])

        demo.load(lambda: Session(), inputs=None, outputs=[session])

    demo.queue().launch()


def main():

    tm_model= _load_model()
    decorator = InternLMChatTemplate()
    _launch_demo(tm_model, decorator)


if __name__ == '__main__':
    main()