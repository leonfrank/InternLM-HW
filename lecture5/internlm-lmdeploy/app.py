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

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
MODEL_PATH='leonfrank/internlm-lmdeploy-personal_assistant'
MODEL_NAME='internlm-chat-7b'

"""A simple web interactive chat demo based on gradio."""

@contextmanager
def get_stop_words():
    from lmdeploy.tokenizer import Tokenizer
    old_func = Tokenizer.indexes_containing_token

    def new_func(self, token):
        indexes = self.encode(token, add_bos=False)
        return indexes

    Tokenizer.indexes_containing_token = new_func
    yield
    Tokenizer.indexes_containing_token = old_func


def _load_model():
    """Load preprocessor and llm inference engine."""
    llm_ckpt = MODEL_PATH
    with get_stop_words():
        model = TurboMind.from_pretrained(llm_ckpt, model_name=MODEL_NAME)
    return model


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



def _launch_demo( model):


    def chat(
        chatbot,
        session,
        request_output_len=512,
    ):
        """Chat with AI assistant."""
        generator = model.create_instance()
        history = session._message
        sequence_start = len(history) == 1
        seed = random.getrandbits(64) if sequence_start else None
        input_ids, features, ranges = preprocessor.prepare_query(
            history[-1][0], sequence_start)

        if len(input_ids
               ) + session.step + request_output_len > model.model.session_len:
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
                response = model.tokenizer.decode(res.tolist(),
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
        generator = model.create_instance()
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


    def predict(_query, _chatbot, _task_history):
        print(f"User: {_parse_text(_query)}")
        _chatbot.append((_parse_text(_query), ""))
        full_response = ""

        for response in model.chat(tokenizer, _query, history=_task_history, generation_config=config):
            _chatbot[-1] = (_parse_text(_query), _parse_text(response))

            yield _chatbot
            full_response = _parse_text(response)

        print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"InternLM: {_parse_text(full_response)}")

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        _gc()
        return _chatbot

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="imgs/汪小姐.png" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>InternLM Bot</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on InternLM \
(本WebUI基于InternLM打造，实现聊天机器人功能。)</center>""")

        chatbot = gr.Chatbot(label='InternLM', elem_classes="control-height")
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            empty_btn = gr.Button("🧹 Clear History (清除历史)")
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

    demo.launch()


def main():

    model= _load_model()
    _launch_demo(model)


if __name__ == '__main__':
    main()