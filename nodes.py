import base64
import io
from PIL import Image
import numpy as np
from openai import OpenAI
import anthropic

class ImageDescriptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_token": ("INT", {"default": 1024}),
                "openai_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
                "model": (["gpt-4-vision Low", "gpt-4-vision High"], {"default": "gpt-4-vision Low"}),
                "prompt": ("STRING", {"multiline": True, "default": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "AppleBotzz/Image/Description"

    def image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def set_system_message(self, sysmsg):
        return [{
            "role": "system",
            "content": sysmsg
        }]

    def describe_image(self, image, max_token, openai_api_key, endpoint, model, prompt):
        try:
            image = image[0]
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if not openai_api_key:
                openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return "OpenAI API key is required for GPT-4 Vision."

            client = OpenAI(api_key=openai_api_key, base_url=endpoint)
            processed_image = self.image_to_base64(img)
            detail = "low" if model == "gpt-4-vision Low" else "high"
            system_message = self.set_system_message("You are GPT-4.")
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=system_message + [
                    {
                        "role": "user",
                        "content": [{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{processed_image}", "detail": detail}
                        }]
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_token
            )
            description = response.choices[0].message.content
            print(f"Descrp[tion: {description}")
            return (description,)
        except Exception as e:
            return (f"Error: {str(e)}",)


class OpenAIChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_token": ("INT", {"default": 1024}),
                "openai_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
                "model": (["gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-4-0125-preview", "gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4-0613"], {"default": "gpt-3.5-turbo"}),
                "prompt": ("STRING", {"multiline": True, "default": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"
    CATEGORY = "AppleBotzz/Chat"
    
    def set_system_message(self, sysmsg):
        return [{
            "role": "system",
            "content": sysmsg
        }]

    def chat(self, max_token, openai_api_key, endpoint, model, prompt):
        try:

            if not openai_api_key:
                openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return "OpenAI API key is required for GPT-4 Vision."

            client = OpenAI(api_key=openai_api_key, base_url=endpoint)
            system_message = self.set_system_message("You are GPT-4.")
            response = client.chat.completions.create(
                model=model,
                messages=system_message + [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_token
            )
            description = response.choices[0].message.content
            print(f"Descrp[tion: {description}")
            return (description,)
        except Exception as e:
            return (f"Error: {str(e)}",)

class ClaudeImageDescriptionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_token": ("INT", {"default": 1024}),
                "claude_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.anthropic.com"}),
                "model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-opus-20240229"}),
                "prompt": ("STRING", {"multiline": True, "default": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "describe_image"
    CATEGORY = "AppleBotzz/Image/Description"

    def image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def describe_image(self, image, max_token, claude_api_key, endpoint, model, prompt):
        try:
            image = image[0]
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if not claude_api_key:
                return "Claude API key is required for Claude models."

            client = anthropic.Anthropic(api_key=claude_api_key, base_url=endpoint)
            processed_image = self.image_to_base64(img)
            message = client.messages.create(
                model=model,
                max_tokens=max_token,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": processed_image,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ],
                    }
                ],
            )
            description = message.content[0].text
            print(description)
            return (description,)
        except Exception as e:
            return (f"Error: {str(e)}",)
            
class ClaudeChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_token": ("INT", {"default": 1024}),
                "claude_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.anthropic.com"}),
                "model": (["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"], {"default": "claude-3-opus-20240229"}),
                "prompt": ("STRING", {"multiline": True, "default": "As an AI image tagging expert, please provide precise tags for these images to enhance CLIP model's understanding of the content. Employ succinct keywords or phrases, steering clear of elaborate sentences and extraneous conjunctions. Prioritize the tags by relevance. Your tags should capture key elements such as the main subject, setting, artistic style, composition, image quality, color tone, filter, and camera specifications, and any other tags crucial for the image. When tagging photos of people, include specific details like gender, nationality, attire, actions, pose, expressions, accessories, makeup, composition type, age, etc. For other image categories, apply appropriate and common descriptive tags as well. Recognize and tag any celebrities, well-known landmark or IPs if clearly featured in the image. Your tags should be accurate, non-duplicative, and within a 20-75 word count range. These tags will use for image re-creation, so the closer the resemblance to the original image, the better the tag quality. Tags should be comma-separated. Exceptional tagging will be rewarded with $10 per image."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "chat"
    CATEGORY = "AppleBotzz/Chat"

    def chat(self, max_token, claude_api_key, endpoint, model, prompt):
        try:

            if not claude_api_key:
                return "Claude API key is required for Claude models."

            client = anthropic.Anthropic(api_key=claude_api_key, base_url=endpoint)
            message = client.messages.create(
                model=model,
                max_tokens=max_token,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            description = message.content[0].text
            print(description)
            return (description,)
        except Exception as e:
            return (f"Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "GPT4_VISION": ImageDescriptionNode,
    "GPT4_CHAT": OpenAIChatNode,
    "CLAUDE_VISION": ClaudeImageDescriptionNode,
    "CLAUDE_CHAT": ClaudeChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT4_VISION" : "GPT-4V Image Chat",
    "GPT4_CHAT" : "OpenAI Chat",
    "CLAUDE_VISION": "Claude-3 Image Chat",
    "CLAUDE_CHAT": "Claude-3 Chat",
}