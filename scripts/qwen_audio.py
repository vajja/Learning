import torch
from qwen_asr import Qwen3ASRModel


class QwenSpeech2Text:
    __inistance = None

    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = Qwen3ASRModel.from_pretrained(
                "Qwen/Qwen3-ASR-0.6B",
                dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                max_inference_batch_size=32,
                # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
                max_new_tokens=256,  # Maximum number of tokens to generate. Set a larger value for long audio input.
            )
            print(f"Successfully loaded Qwen3-ASR-0.6B")
        except Exception as ex:
            print(f"Error while loading qwen-asr")

        return

    @staticmethod
    def get_instance():
        if QwenSpeech2Text.__inistance is None:
            QwenSpeech2Text.__instance = QwenSpeech2Text()
        print("returning Qwen singletong")
        return QwenSpeech2Text.__instance

    def speech_text(self, file_name):
        try:
            results = self.model.transcribe(
                audio=file_name,
                language="English",  # set "English" to force the language
            )
            return results[0].text
        except Exception as ex:
            print(f"Error while converting speech to text")
        return ""