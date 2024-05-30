from transformers import GPT2Tokenizer
from vllm.logger import init_logger


logger=init_logger(__name__)

class DataMocking:
    def __init__(self):
        self.tokenizer=GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        self.device='cpu'
            
        
    def tokenize(self,input_string):
        try:
            model_inputs = self.tokenizer([input_string], return_tensors="pt")
            token_num = len(model_inputs.input_ids[0])
            return token_num
        except Exception as e:
            logger.error("Tokenization failed: %s", str(e))
            return -1
            
            
    @staticmethod
    def remove_last_word(input_string):
        words = input_string.split()
        if words:
            words.pop()
        return ' '.join(words)

    @staticmethod
    def duplicate_last_word(input_string):
        words = input_string.split()
        if words:
            last_word = words[-1]
            words.append(last_word)
        return ' '.join(words)

    def adjust_to_limit(self, _input_string, _target_length):
        try:
            attempt_tokenize = (self.tokenizer([_input_string], return_tensors="pt")
                                    .to(self.device))
            token_num = len(attempt_tokenize.input_ids[0])
            while token_num > _target_length:
                _input_string = self.remove_last_word(_input_string)
                attempt_tokenize = (self.tokenizer([_input_string], return_tensors="pt")
                                    .to(self.device))
                token_num = len(attempt_tokenize.input_ids[0])

            while token_num < _target_length:
                _input_string = self.duplicate_last_word(_input_string)
                attempt_tokenize = (self.tokenizer([_input_string], return_tensors="pt")
                                    .to(self.device))
                token_num = len(attempt_tokenize.input_ids[0])

            return _input_string

        except Exception as e:
            logger.error("Tokenization failed: %s", str(e))
            return _input_string  # Changed from `return input` to `return _input_string`

    @staticmethod
    def calculate_num_repeats(phrase, token_len):
        words = phrase.split()  # Split the phrase into words
        phrase_length = len(words)  # Calculate the length of the phrase in words
        num_repeats = token_len // phrase_length  # Calculate the number of repeats needed
        remainder = token_len % phrase_length  # Calculate the remainder
        return num_repeats, remainder

    def create_prompt(self, prompt_token_len):
        prompt_phrase = "please respond as long as possible"
        prompt_num_repeats, prompt_remainder = self.calculate_num_repeats(prompt_phrase, prompt_token_len)
        prompt = (prompt_phrase + " ") * prompt_num_repeats
        if prompt_remainder > 0:
            prompt += " ".join(prompt_phrase.split()[:prompt_remainder])
        return self.adjust_to_limit(prompt, prompt_token_len)