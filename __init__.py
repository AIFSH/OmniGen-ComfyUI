import sys
import folder_paths
import os.path as osp
now_dir = osp.dirname(__file__)
aifsh_dir = osp.join(folder_paths.models_dir,"AIFSH")
sys.path.append(now_dir)

from huggingface_hub import snapshot_download
omnigen_dir = osp.join(aifsh_dir,"Shitao/OmniGen-v1")
tmp_dir = osp.join(now_dir, "tmp")
import os
import shutil
import torch
import tempfile
import numpy as np
from PIL import Image
from OmniGen import OmniGenPipeline

class OmniGenNode:
    def __init__(self):
        if not osp.exists(osp.join(omnigen_dir,"model.safetensors")):
            snapshot_download("Shitao/OmniGen-v1",local_dir=omnigen_dir)
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt_text":("TEXT",{
                    "tooltip":"you only need image_1, text will auto be <img><|image_1|></img>"
                }),
                "height":("INT",{
                    "default":1024,
                    "min":128,
                    "max":2048,
                    "step":16,
                    "display":"slider"
                }),
                "width":("INT",{
                    "default":1024,
                    "min":128,
                    "max":2048,
                    "step":16,
                    "display":"slider"
                }),
                "num_inference_steps":("INT",{
                    "default":50,
                    "min":1,
                    "max":100,
                    "step":1,
                    "display":"slider"
                }),
                "guidance_scale":("FLOAT",{
                    "default":2.5,
                    "min":1.0,
                    "max":5.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "img_guidance_scale":("FLOAT",{
                    "default":1.6,
                    "min":1.0,
                    "max":2.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "max_input_image_size":("INT",{
                    "default":1024,
                    "min":128,
                    "max":2048,
                    "step":16,
                    "display":"slider"
                }),
                "separate_cfg_infer":("BOOLEAN",{
                    "default":True,
                    "tooltip":"Whether to use separate inference process for different guidance. This will reduce the memory cost."
                }),
                "offload_model":("BOOLEAN",{
                    "default":False,
                    "tooltip":"Offload model to CPU, which will significantly reduce the memory cost but slow down the generation speed. You can cancle separate_cfg_infer and set offload_model=True. If both separate_cfg_infer and offload_model be True, further reduce the memory, but slowest generation"
                }),
                "use_input_image_size_as_output":("BOOLEAN",{
                    "default":False,
                    "tooltip":"Automatically adjust the output image size to be same as input image size. For editing and controlnet task, it can make sure the output image has the same size with input image leading to better performance"
                }),
                "seed":("INT",{
                    "default":42
                })
            },
            "optional":{
                "image_1":("IMAGE",),
                "image_2":("IMAGE",),
                "image_3":("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_OmniGen"

    def save_input_img(self,image):
        with tempfile.NamedTemporaryFile(suffix=".png",delete=False,dir=tmp_dir) as f:
            img_np = image.numpy()[0]*255
            img_pil = Image.fromarray(img_np.astype(np.uint8))
            img_pil.save(f.name)
        return f.name

    def gen(self,prompt_text,height,width,num_inference_steps,guidance_scale,
            img_guidance_scale,max_input_image_size,separate_cfg_infer,offload_model,
            use_input_image_size_as_output,seed,image_1=None,image_2=None,image_3=None):
        if not hasattr( self, "omnigen_pipe" ):
            setattr( self, "omnigen_pipe", OmniGenPipeline.from_pretrained(omnigen_dir) )
        pipe = getattr( self, "omnigen_pipe" )
        input_images = []
        os.makedirs(tmp_dir,exist_ok=True)
        if image_1 is not None:
            input_images.append(self.save_input_img(image_1))
            prompt_text = prompt_text.replace("image_1","<img><|image_1|></img>")
        
        if image_2 is not None:
            input_images.append(self.save_input_img(image_2))
            prompt_text = prompt_text.replace("image_2","<img><|image_2|></img>")
        
        if image_3 is not None:
            input_images.append(self.save_input_img(image_3))
            prompt_text = prompt_text.replace("image_3","<img><|image_3|></img>")
        
        if len(input_images) == 0:
            input_images = None
            
        print(prompt_text)
        output = pipe(
            prompt=prompt_text,
            input_images=input_images,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            img_guidance_scale=img_guidance_scale,
            num_inference_steps=num_inference_steps,
            separate_cfg_infer=separate_cfg_infer, 
            use_kv_cache=True,
            offload_kv_cache=True,
            offload_model=offload_model,
            use_input_image_size_as_output=use_input_image_size_as_output,
            seed=seed,
            max_input_image_size=max_input_image_size,
        )
        img = np.array(output[0]) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        # print(img.shape)
        shutil.rmtree(tmp_dir)
        return (img,)


NODE_CLASS_MAPPINGS = {
    "OmniGenNode": OmniGenNode
}
