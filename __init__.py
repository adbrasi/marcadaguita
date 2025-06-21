import torch
import numpy as np
from PIL import Image, ImageOps
import os

class WatermarkNode:
    """
    Custom node para adicionar marca d'água em imagens no ComfyUI
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "watermark_path": ("STRING", {"default": "", "multiline": False}),
                "position": (["top_left", "top_center", "top_right", 
                           "center_left", "center", "center_right",
                           "bottom_left", "bottom_center", "bottom_right"], 
                           {"default": "bottom_right"}),
                "opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "margin_x": ("INT", {"default": 20, "min": 0, "max": 500, "step": 1}),
                "margin_y": ("INT", {"default": 20, "min": 0, "max": 500, "step": 1}),
                "offset_x": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
                "offset_y": ("INT", {"default": 0, "min": -500, "max": 500, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_watermark"
    CATEGORY = "image/postprocessing"
    
    def apply_watermark(self, image, watermark_path, position, opacity, scale, 
                       margin_x, margin_y, offset_x, offset_y):
        """
        Aplica a marca d'água na imagem
        """
        try:
            # Converter tensor para PIL Image
            if isinstance(image, torch.Tensor):
                # ComfyUI usa formato [batch, height, width, channels]
                img_array = image[0].cpu().numpy()
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                base_image = Image.fromarray(img_array)
            else:
                base_image = image
            
            # Verificar se o arquivo de marca d'água existe
            if not os.path.exists(watermark_path):
                print(f"Arquivo de marca d'água não encontrado: {watermark_path}")
                # Retornar imagem original sem alterações
                return (image,)
            
            # Carregar marca d'água
            try:
                watermark = Image.open(watermark_path).convert("RGBA")
            except Exception as e:
                print(f"Erro ao carregar marca d'água: {e}")
                return (image,)
            
            # Obter dimensões da imagem base
            base_width, base_height = base_image.size
            
            # Calcular tamanho da marca d'água baseado na escala
            # Usar a menor dimensão como referência para manter proporção
            min_dimension = min(base_width, base_height)
            watermark_size = int(min_dimension * scale)
            
            # Redimensionar marca d'água mantendo aspect ratio
            watermark_ratio = watermark.width / watermark.height
            if watermark.width > watermark.height:
                new_width = watermark_size
                new_height = int(watermark_size / watermark_ratio)
            else:
                new_height = watermark_size
                new_width = int(watermark_size * watermark_ratio)
            
            watermark = watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Calcular posição baseada no preset selecionado
            pos_x, pos_y = self._calculate_position(
                base_width, base_height, 
                watermark.width, watermark.height,
                position, margin_x, margin_y, offset_x, offset_y
            )
            
            # Garantir que a marca d'água não seja cortada
            pos_x, pos_y = self._ensure_watermark_fits(
                base_width, base_height,
                watermark.width, watermark.height,
                pos_x, pos_y
            )
            
            # Converter imagem base para RGBA se necessário
            if base_image.mode != 'RGBA':
                base_image = base_image.convert('RGBA')
            
            # Aplicar opacidade à marca d'água
            if opacity < 1.0:
                # Criar uma cópia da marca d'água com opacidade ajustada
                watermark_with_opacity = watermark.copy()
                alpha = watermark_with_opacity.split()[-1]  # Canal alpha
                alpha = alpha.point(lambda p: int(p * opacity))
                watermark_with_opacity.putalpha(alpha)
                watermark = watermark_with_opacity
            
            # Criar uma imagem temporária para composição
            temp_img = Image.new('RGBA', base_image.size, (0, 0, 0, 0))
            temp_img.paste(watermark, (pos_x, pos_y), watermark)
            
            # Compor as imagens
            result = Image.alpha_composite(base_image, temp_img)
            
            # Converter de volta para RGB se a imagem original era RGB
            if image[0].shape[-1] == 3:  # Se a imagem original tinha 3 canais
                result = result.convert('RGB')
            
            # Converter de volta para tensor do ComfyUI
            result_array = np.array(result).astype(np.float32) / 255.0
            result_tensor = torch.from_numpy(result_array).unsqueeze(0)
            
            return (result_tensor,)
            
        except Exception as e:
            print(f"Erro ao aplicar marca d'água: {e}")
            return (image,)  # Retornar imagem original em caso de erro
    
    def _calculate_position(self, base_width, base_height, wm_width, wm_height,
                          position, margin_x, margin_y, offset_x, offset_y):
        """
        Calcula a posição da marca d'água baseada no preset selecionado
        """
        positions = {
            "top_left": (margin_x, margin_y),
            "top_center": ((base_width - wm_width) // 2, margin_y),
            "top_right": (base_width - wm_width - margin_x, margin_y),
            "center_left": (margin_x, (base_height - wm_height) // 2),
            "center": ((base_width - wm_width) // 2, (base_height - wm_height) // 2),
            "center_right": (base_width - wm_width - margin_x, (base_height - wm_height) // 2),
            "bottom_left": (margin_x, base_height - wm_height - margin_y),
            "bottom_center": ((base_width - wm_width) // 2, base_height - wm_height - margin_y),
            "bottom_right": (base_width - wm_width - margin_x, base_height - wm_height - margin_y)
        }
        
        base_x, base_y = positions.get(position, positions["bottom_right"])
        
        # Aplicar offsets
        final_x = base_x + offset_x
        final_y = base_y + offset_y
        
        return final_x, final_y
    
    def _ensure_watermark_fits(self, base_width, base_height, wm_width, wm_height, pos_x, pos_y):
        """
        Garante que a marca d'água não seja cortada, ajustando a posição se necessário
        """
        # Ajustar X
        if pos_x < 0:
            pos_x = 0
        elif pos_x + wm_width > base_width:
            pos_x = base_width - wm_width
        
        # Ajustar Y
        if pos_y < 0:
            pos_y = 0
        elif pos_y + wm_height > base_height:
            pos_y = base_height - wm_height
        
        return pos_x, pos_y


# Registrar o node
NODE_CLASS_MAPPINGS = {
    "WatermarkNode": WatermarkNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WatermarkNode": "marcadagua bumbumzin sujo"
}