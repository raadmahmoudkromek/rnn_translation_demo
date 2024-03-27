# function to generate output sequence using greedy algorithm
import torch

from src.data_processing import generate_square_subsequent_mask, DataProcessor
from src.device import DEVICE


def greedy_decode(model, src, src_mask, max_len, start_symbol, data_processor):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == data_processor.eos_id:
            break
    return ys

def translate(model: torch.nn.Module, src_sentence: str, data_processor: DataProcessor):
    model.eval()
    src = data_processor.lang_a_text_transform(src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=data_processor.bos_id,
        data_processor=data_processor).flatten()
    return " ".join(data_processor.vocab_b.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace(
        "<eos>", "")