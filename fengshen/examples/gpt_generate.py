import imp
import torch
from torchsummary import summary
from transformers import GPT2Tokenizer,GPT2LMHeadModel,BertTokenizer

hf_model_path = '/Users/wuziwei/data/NLPmodel/gpt2-chinese-cluecorpussmall'

tokenizer = BertTokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)

# 查看模型结构
# print(model)

inputs = tokenizer('这是很久之前的事情了',return_tensors='pt')

# 查看生产方法的详细参数
# https://huggingface.co/docs/transformers/master/en/main_classes/model#transformers.generation_utils.GenerationMixin.generate
# print(help(model.generate))
# generation
generation_output = model.generate(**inputs,return_dict_in_generate=True,output_scores=True,max_length=100,max_new_tokens=30,do_sample=True,num_beams=10)
"""
generate:
 |      max_length,max_new_tokens:二选一，后者是不包括输入的最大输出长度。
 |      do_sample=True:是否采样，
 |
 |
 |
 |
 |
 |
 |
 |
generation_output:
 |      sequences (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
 |          The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
 |          shorter if all batches finished early due to the :obj:`eos_token_id`.
 |      scores (:obj:`tuple(torch.FloatTensor)` `optional`, returned when ``output_scores=True`` is passed or when ``config.output_scores=True``):
 |          Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
 |          at each generation step. :obj:`(max_length-input_ids.shape[-1],)`-shaped tuple of :obj:`torch.FloatTensor`
 |          with each tensor of shape :obj:`(batch_size, config.vocab_size)`).
 |      attentions (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
 |          Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
 |          :obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_heads, generated_length, sequence_length)`.
 |      hidden_states (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
 |          Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
 |          :obj:`torch.FloatTensor` of shape :obj:`(batch_size, generated_length, hidden_size)`.
 """

print(generation_output.sequences)
print(tokenizer.decode(generation_output.sequences[0]))

