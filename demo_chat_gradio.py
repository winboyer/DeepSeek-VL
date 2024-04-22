# -*- coding: utf-8 -*-

import argparse
import os
import sys
from PIL import Image
from threading import Thread
import torch
from transformers import TextIteratorStreamer

from deepseek_vl.utils.io import load_pretrained_model

import gradio as gr
from functools import partial


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-vl-7b-chat",
                        help="the huggingface model name or the local path of the downloaded huggingface model.")
    parser.add_argument("--max_gen_len", type=int, default=512)
    args = parser.parse_args()

    return args


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image


def get_help_message(image_token):
    help_msg = (
        f"\t\t DeepSeek-VL-Chat is a chatbot that can answer questions based on the given image. Enjoy it! \n"
        f"Usage: \n"
        f"    1. type `exit` to quit. \n"
        f"    2. type `{image_token}` to indicate there is an image. You can enter multiple images, "
        f"e.g '{image_token} is a dot, {image_token} is a cat, and what is it in {image_token}?'. "
        f"When you type `{image_token}`, the chatbot will ask you to input image file path. \n"
        f"    4. type `help` to get the help messages. \n"
        f"    5. type `new` to start a new conversation. \n"
        f"    Here is an example, you can type: '<image_placeholder>Describe the image.'\n"
    )

    return help_msg


@torch.inference_mode()
def response(conv, pil_images, tokenizer, vl_chat_processor, vl_gpt, generation_config):

    prompt = conv.get_prompt()
    prepare_inputs = vl_chat_processor.__call__(
        prompt=prompt,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_config["inputs_embeds"] = inputs_embeds
    generation_config["attention_mask"] = prepare_inputs.attention_mask
    generation_config["streamer"] = streamer

    thread = Thread(target=vl_gpt.language_model.generate, kwargs=generation_config)
    thread.start()

    yield from streamer


def get_user_input(hint: str):
    user_input = ""
    while user_input == "":
        try:
            user_input = input(f"{hint}")
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            user_input = "exit"

    return user_input


def chat_inference(tokenizer, vl_chat_processor, vl_gpt, temperature, top_p, repetition_penalty):
    
    image_token = vl_chat_processor.image_token
    generation_config = dict(
        pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
        bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
        eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
        use_cache=True,
    )
    if temperature > 0:
        generation_config.update({
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
        })
    else:
        generation_config.update({"do_sample": False})


    pil_images = []
    conv = vl_chat_processor.new_chat_template()
    roles = conv.roles

    while True:

        # get user input
        user_input = get_user_input(f"{roles[0]} [{image_token} indicates an image]: ")

        elif user_input == "new":
            os.system("clear")
            pil_images = []
            conv = vl_chat_processor.new_chat_template()
            torch.cuda.empty_cache()
            print("New conversation started.")

        else:
            conv.append_message(conv.roles[0], user_input)
            conv.append_message(conv.roles[1], None)

            # check if the user input is an image token
            num_images = user_input.count(image_token)
            cur_img_idx = 0

            while cur_img_idx < num_images:
                try:
                    image_file = input(f"({cur_img_idx + 1}/{num_images}) Input the image file path: ")

                except KeyboardInterrupt:
                    print()
                    continue

                except EOFError:
                    image_file = None

                if image_file and os.path.exists(image_file):
                    pil_image = load_image(image_file)
                    pil_images.append(pil_image)
                    cur_img_idx += 1

                elif image_file == "exit":
                    print("Chat program exited.")
                    sys.exit(0)

                else:
                    print(f"File error, `{image_file}` does not exist. Please input the correct file path.")

            # get the answer by the model's prediction
            answer = ""
            answer_iter = response(conv, pil_images, tokenizer, vl_chat_processor, vl_gpt, generation_config)
            sys.stdout.write(f"{conv.roles[1]}: ")
            for char in answer_iter:
                answer += char
                sys.stdout.write(char)
                sys.stdout.flush()

            sys.stdout.write("\n")
            sys.stdout.flush()
            conv.update_last_message(answer)
            # conv.messages[-1][-1] = answer


def main(args):

    tokenizer, vl_chat_processor, vl_gpt = load_pretrained_model(args.model_path)
    

    with gr.Blocks(title="DeepSeek-VL") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>DeepSeek-VL</center></h1>')
        with gr.Row():
            temperature = gr.Slider(minimum=0,
                                maximum=1,
                                value=0.2,
                                step=0.01,
                                interactive=True,
                                label='temperature value')
            top_p = gr.Slider(minimum=0,
                                maximum=1,
                                value=0.95,
                                step=0.01,
                                interactive=True,
                                label='top_p value')
            repetition_penalty = gr.Slider(minimum=0,
                                maximum=2,
                                value=1.1,
                                step=0.01,
                                interactive=True,
                                label='repetition_penalty value')
        with gr.Tab("Image"):
            input_text_image = gr.Textbox(label='Enter the question you want to know',
                                    value='What is the image about',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_image_file = gr.Image(type='pil', label='Input Image')
                    input_image_file = gr.Image(type='filepath', label='Input Image')      
                with gr.Column(scale=7):
                    result_text_image = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            with gr.Row():
                image_submit = gr.Button('Submit')
                image_clear = gr.Button('Clear')
        with gr.Tab("Video"):
            input_text_video = gr.Textbox(label='Enter the question you want to know',
                                    value='What is the video about',
                                    elem_id='textbox')
            with gr.Row():
                with gr.Column(scale=4):
                    # input_video_file = gr.File(label="Input Video", file_types=[".mp4",".ts",".avi",".mpg",".mpeg",".rm",".rmvb",".mov",".wmv"])
                    input_video_file = gr.Video(label="Input Video")
                with gr.Column(scale=7):
                    result_text_video = gr.components.Chatbot(label='Conversation History', 
                                                # label='Multi-round conversation History', 
                                                value=[("", "Hi, What do you want to know about?")], 
                                                height=500)
            with gr.Row():
                video_submit = gr.Button('Submit')
                video_clear = gr.Button('Clear')
        
        video_submit.click(partial(chat_inference, tokenizer, vl_chat_processor, vl_gpt),
                                [input_video_file, input_text_video, max_n_frames],
                                [result_text_video])
        video_clear.click(fn=video_clear_fn, outputs=[input_video_file, input_text_video, result_text_image])
        image_submit.click(partial(chat_inference, tokenizer, vl_chat_processor, vl_gpt),
                                [input_image_file, input_text_image, max_n_frames],
                                [result_text_image])
        image_clear.click(lambda: [[], '', ''], None,
                                [input_image_file, input_text_image, result_text_image])

        demo.queue(concurrency_count=5)
        demo.launch(server_name='0.0.0.0', server_port=8080)
        # demo.launch(server_name='0.0.0.0', server_port=args.server_port)

    chat(args, tokenizer, vl_chat_processor, vl_gpt, generation_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-vl-7b-chat",
                        help="the huggingface model name or the local path of the downloaded huggingface model.")
    parser.add_argument("--max_gen_len", type=int, default=512)
    args = parser.parse_args()
    main(args)

